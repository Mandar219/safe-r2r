from typing import List
from safe_r2r.retrieval.faiss_retriever import RetrievedDoc

def build_rag_prompt(question: str, docs: List[RetrievedDoc]) -> str:
    ctx = "\n\n".join(
        [f"[Doc {i+1}] {d.title}\n{d.text}" for i, d in enumerate(docs)]
    )
    return f"""You are a question answering system. Use the documents to answer.
If the documents do not contain enough information, say "Insufficient evidence".

Documents:
{ctx}

Question: {question}
Answer:"""