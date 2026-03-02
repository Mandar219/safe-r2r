import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    title: str
    text: str

class FaissRetriever:
    def __init__(self, index_path: str, ids_path: str, meta_path: str, doc_lookup_path: str):
        self.index = faiss.read_index(index_path)

        with open(ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        with open(doc_lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)

        self.model = SentenceTransformer(self.meta["embedding_model"])
        self.normalize = bool(self.meta["normalize_embeddings"])

    def search(self, query: str, top_k: int) -> List[RetrievedDoc]:
        q_emb = self.model.encode([query], normalize_embeddings=self.normalize)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, idxs = self.index.search(q_emb, top_k)
        out: List[RetrievedDoc] = []

        for score, idx in zip(scores[0], idxs[0]):
            doc_id = self.doc_ids[int(idx)]
            doc = self.lookup[doc_id]
            out.append(RetrievedDoc(
                doc_id=doc_id,
                score=float(score),
                title=doc["title"],
                text=doc["text"]
            ))
        return out