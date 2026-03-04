from dataclasses import dataclass
from typing import Dict, List, Optional

from safe_r2r.retrieval.faiss_retriever import FaissRetriever, RetrievedDoc
from safe_r2r.retrieval.reranker import CrossEncoderReranker


@dataclass
class LadderConfig:
    rung_top_k: Dict[int, int]
    # Reranker settings (optional)
    rerank_faiss_top_n: int = 50


class RetrievalLadder:
    """
    Ladder:
      rung 0 -> no retrieval
      rung 1 -> FAISS top_k=5
      rung 2 -> FAISS top_k=10
      rung 3 -> FAISS top_k=15
      rung 4 -> FAISS top_n=50 -> rerank -> keep top_k=15
    """

    def __init__(
        self,
        retriever: FaissRetriever,
        cfg: LadderConfig,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.retriever = retriever
        self.cfg = cfg
        self.reranker = reranker

    def top_k_for_rung(self, rung: int) -> int:
        if rung == 0:
            return 0
        if rung not in self.cfg.rung_top_k:
            raise ValueError(f"Unknown rung={rung}. Valid rungs: {sorted([0] + list(self.cfg.rung_top_k.keys()))}")
        return int(self.cfg.rung_top_k[rung])

    def retrieve(self, question: str, rung: int) -> List[RetrievedDoc]:
        top_k = self.top_k_for_rung(rung)
        if rung == 0 or top_k <= 0:
            return []

        # Simple FAISS-only rungs
        if rung in (1, 2, 3) or self.reranker is None:
            return self.retriever.search(question, top_k=top_k)

        # Rerank rung (4)
        if rung == 4:
            faiss_top_n = int(self.cfg.rerank_faiss_top_n)
            # Stage 1: dense retrieval
            cand_docs = self.retriever.search(question, top_k=faiss_top_n)

            # Stage 2: cross-encoder scoring
            title_text = [(d.title, d.text) for d in cand_docs]
            scores = self.reranker.rerank(question, title_text)

            # Sort by cross-encoder score desc
            scored = list(zip(cand_docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Return top_k with updated score = rerank score
            out: List[RetrievedDoc] = []
            for d, s in scored[:top_k]:
                out.append(RetrievedDoc(
                    doc_id=d.doc_id,
                    score=float(s),
                    title=d.title,
                    text=d.text,
                ))
            return out

        raise ValueError(f"Unhandled rung={rung}")