from typing import List
from safe_r2r.retrieval.faiss_retriever import RetrievedDoc

def build_rag_prompt(
    question: str,
    docs: List[RetrievedDoc],
    *,
    max_doc_tokens: int = 120,
    max_ctx_tokens: int = 1600,
    max_title_tokens: int = 12,
) -> str:
    """
    Strict prompt with truncation to control prompt tokens.

    Truncation strategy:
      1) truncate each doc to max_doc_tokens (word-based proxy for tokens)
      2) then enforce a global max_ctx_tokens across all docs (also word-based proxy)

    Note: This uses whitespace tokens as a proxy. True tokenizer-based counting is
    handled in LLM meta logging; this keeps prompt-building lightweight.
    """

    def trunc_words(text: str, n: int) -> str:
        ws = text.split()
        if len(ws) <= n:
            return text.strip()
        return " ".join(ws[:n]).strip() + " ..."

    remaining = max_ctx_tokens
    doc_blocks = []

    for i, d in enumerate(docs, start=1):
        # title truncation (optional)
        title = trunc_words(d.title.strip(), max_title_tokens)

        # doc text truncation
        text = trunc_words(d.text.strip(), max_doc_tokens)

        # enforce global budget using word count proxy
        block = f"[Doc {i}] {title}\n{text}"
        block_words = len(block.split())

        if block_words > remaining:
            # If we still have room, add a further-truncated version; else stop
            if remaining <= 20:
                break
            # Shrink just the text to fit
            # Reserve some words for header
            header = f"[Doc {i}] {title}\n"
            header_words = len(header.split())
            avail_for_text = max(0, remaining - header_words)
            if avail_for_text <= 10:
                break
            text = trunc_words(d.text.strip(), min(max_doc_tokens, avail_for_text))
            block = header + text
            block_words = len(block.split())

            if block_words > remaining:
                break

        doc_blocks.append(block)
        remaining -= block_words

        if remaining <= 20:
            break

    ctx = "\n\n".join(doc_blocks)

    return (
        "You are a question answering system.\n"
        "Use ONLY the provided documents to answer the question.\n\n"
        "You may think step-by-step internally to ensure correctness, "
        "but you MUST output ONLY the final answer.\n\n"
        "Output Rules (MUST FOLLOW):\n"
        "1) Output ONLY the final answer as a short span (one line).\n"
        "2) Do NOT explain.\n"
        "3) Do NOT add extra words.\n"
        "4) Do NOT include quotes.\n"
        "5) Do NOT repeat the question.\n"
        "6) If yes/no, output exactly: yes OR no\n"
        "7) If the documents do not contain enough information, output exactly: Insufficient evidence\n\n"
        f"Documents:\n{ctx}\n\n"
        f"Question: {question}\n"
        "Final answer:"
    )