from dataclasses import dataclass
from typing import Dict, List, Optional

from safe_r2r.retrieval.faiss_retriever import FaissRetriever, RetrievedDoc
from safe_r2r.retrieval.reranker import CrossEncoderReranker
from safe_r2r.retrieval.compress import ExtractiveCompressor


@dataclass
class LadderConfig:
    rung_top_k: Dict[int, int]
    rerank_faiss_top_n: int = 50


class RetrievalLadder:
    """
    Ladder:
      rung 0 -> no retrieval
      rung 1 -> FAISS top 5 + extractive compression
      rung 2 -> FAISS top 15 + extractive compression
      rung 3 -> FAISS top 50 -> rerank -> top 15 + extractive compression
      rung 4 -> FAISS top 50 -> rerank -> top 15 full docs
    """

    def __init__(
        self,
        retriever: FaissRetriever,
        cfg: LadderConfig,
        reranker: Optional[CrossEncoderReranker] = None,
        compressor: Optional[ExtractiveCompressor] = None,
    ):
        self.retriever = retriever
        self.cfg = cfg
        self.reranker = reranker
        self.compressor = compressor

    def top_k_for_rung(self, rung: int) -> int:
        if rung == 0:
            return 0
        if rung not in self.cfg.rung_top_k:
            raise ValueError(f"Unknown rung={rung}. Valid: {sorted([0] + list(self.cfg.rung_top_k.keys()))}")
        return int(self.cfg.rung_top_k[rung])

    def _maybe_compress(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if self.compressor is None:
            return docs
        out: List[RetrievedDoc] = []
        for d in docs:
            ctext = self.compressor.compress(query, d.text)
            # If compression fails, fall back to original
            out.append(RetrievedDoc(
                doc_id=d.doc_id,
                score=d.score,
                title=d.title,
                text=ctext if ctext else d.text,
            ))
        return out

    def retrieve(self, question: str, rung: int) -> List[RetrievedDoc]:
        top_k = self.top_k_for_rung(rung)
        if rung == 0 or top_k <= 0:
            return []

        # rungs 1 and 2: FAISS only + compression
        if rung in (1, 2):
            docs = self.retriever.search(question, top_k=top_k)
            return self._maybe_compress(question, docs)

        # rungs 3 and 4: FAISS topN -> rerank -> keep topK
        if rung in (3, 4):
            if self.reranker is None:
                raise ValueError("Rung 3/4 requires reranker but reranker is None")

            faiss_top_n = int(self.cfg.rerank_faiss_top_n)
            cand_docs = self.retriever.search(question, top_k=faiss_top_n)

            title_text = [(d.title, d.text) for d in cand_docs]
            scores = self.reranker.rerank(question, title_text)

            scored = list(zip(cand_docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            top_docs: List[RetrievedDoc] = []
            for d, s in scored[:top_k]:
                top_docs.append(RetrievedDoc(
                    doc_id=d.doc_id,
                    score=float(s),  # reranker score
                    title=d.title,
                    text=d.text,
                ))

            # rung 3: compressed; rung 4: full
            if rung == 3:
                return self._maybe_compress(question, top_docs)
            return top_docs

        raise ValueError(f"Unhandled rung={rung}")