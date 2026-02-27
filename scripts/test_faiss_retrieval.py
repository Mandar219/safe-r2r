import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from safe_r2r.utils.io import load_yaml

def main():
    cfg = load_yaml("configs/default.yaml")
    ds_name = cfg["dataset"]["name"]
    ds_cfg  = cfg["dataset"]["config"]
    artifacts_dir = cfg["paths"]["artifacts"]

    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}.index"
    ids_path   = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_doc_ids.json"
    meta_path  = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_meta.json"

    index = faiss.read_index(index_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = SentenceTransformer(meta["embedding_model"])

    q = "Were Scott Derrickson and Ed Wood of the same nationality?"
    q_emb = model.encode([q], normalize_embeddings=meta["normalize_embeddings"])
    q_emb = np.asarray(q_emb, dtype=np.float32)

    top_k = 5
    scores, idxs = index.search(q_emb, top_k)

    print("Query:", q)
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        print(rank, "doc_id:", doc_ids[int(idx)], "score:", float(score))

if __name__ == "__main__":
    main()