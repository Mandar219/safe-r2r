# src/safe_r2r/retrieval/compress.py
from dataclasses import dataclass
from typing import List
import re
import numpy as np

from sentence_transformers import SentenceTransformer


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


@dataclass
class CompressionConfig:
    embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    max_sents_per_doc: int = 4
    max_words_per_sent: int = 35
    max_words_per_doc: int = 140


class ExtractiveCompressor:
    """
    Extractive sentence-level compressor.
    Scores each sentence by cosine similarity to the query embedding.
    Keeps top-k sentences and applies hard word caps.
    """

    def __init__(self, cfg: CompressionConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedder_model_name)

    @staticmethod
    def _clean_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
        # Remove extremely short junk
        sents = [s for s in sents if len(s.split()) >= 3]
        return sents

    @staticmethod
    def _cap_words(text: str, max_words: int) -> str:
        ws = text.split()
        if len(ws) <= max_words:
            return text.strip()
        return " ".join(ws[:max_words]).strip() + " ..."

    def compress(self, query: str, doc_text: str) -> str:
        sents = self._clean_sentences(doc_text)
        if not sents:
            return ""

        # Cap sentence length before embedding (keeps compute + tokens down)
        sents_capped = [self._cap_words(s, self.cfg.max_words_per_sent) for s in sents]

        # Encode query + sentences (normalize_embeddings=True gives cosine via dot)
        q_emb = self.model.encode([query], normalize_embeddings=True)
        s_emb = self.model.encode(
            sents_capped,
            normalize_embeddings=True,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
        )

        q = np.asarray(q_emb[0], dtype=np.float32)
        S = np.asarray(s_emb, dtype=np.float32)

        scores = S @ q  # cosine similarity due to normalization
        top_idx = np.argsort(-scores)[: self.cfg.max_sents_per_doc]

        # Keep in original order to read naturally (optional but helps LLM)
        top_idx_sorted = sorted(top_idx.tolist())

        kept = [sents_capped[i] for i in top_idx_sorted]
        out = " ".join(kept).strip()
        out = self._cap_words(out, self.cfg.max_words_per_doc)

        return out