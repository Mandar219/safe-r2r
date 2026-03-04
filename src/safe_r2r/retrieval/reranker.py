from dataclasses import dataclass
from typing import List, Tuple

from sentence_transformers import CrossEncoder


@dataclass
class RerankerConfig:
    model_name: str
    batch_size: int = 32


class CrossEncoderReranker:
    """
    Cross-encoder reranker: scores (query, doc_text) pairs.
    Returns docs sorted by descending score.
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        self.model = CrossEncoder(cfg.model_name)

    def rerank(self, query: str, docs: List[Tuple[str, str]]) -> List[float]:
        """
        docs: List[(title, text)]
        returns: List[float] scores aligned with docs
        """
        pairs = [(query, f"{title}\n{text}") for title, text in docs]
        scores = self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]