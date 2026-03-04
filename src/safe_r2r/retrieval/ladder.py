from dataclasses import dataclass
from typing import Dict, List, Optional

from safe_r2r.retrieval.faiss_retriever import FaissRetriever, RetrievedDoc


@dataclass
class LadderConfig:
    rung_top_k: Dict[int, int]


class RetrievalLadder:
    """
    Minimal ladder:
      rung 0 -> no retrieval
      rung 1 -> faiss top_k=5
      rung 2 -> faiss top_k=10
      rung 3 -> faiss top_k=15
    """

    def __init__(self, retriever: FaissRetriever, cfg: LadderConfig):
        self.retriever = retriever
        self.cfg = cfg

    def top_k_for_rung(self, rung: int) -> int:
        if rung == 0:
            return 0
        if rung not in self.cfg.rung_top_k:
            raise ValueError(f"Unknown rung={rung}. Valid: 0..{max(self.cfg.rung_top_k.keys())}")
        return int(self.cfg.rung_top_k[rung])

    def retrieve(self, question: str, rung: int) -> List[RetrievedDoc]:
        top_k = self.top_k_for_rung(rung)
        if top_k <= 0:
            return []
        return self.retriever.search(question, top_k=top_k)