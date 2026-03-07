from typing import List
from safe_r2r.retrieval.faiss_retriever import RetrievedDoc

def build_rag_prompt(question: str, docs: List[RetrievedDoc]) -> str:
    """
    Strict prompt to force short, span-only outputs.
    This is crucial for HotpotQA EM/F1.
    """
    ctx = "\n\n".join(
        [f"[Doc {i+1}] {d.title}\n{d.text}" for i, d in enumerate(docs)]
    )

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