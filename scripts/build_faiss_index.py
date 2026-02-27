import json
import time
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from safe_r2r.utils.io import load_yaml, ensure_dir

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    cfg = load_yaml("configs/default.yaml")
    processed_dir = cfg["paths"]["data_processed"]
    artifacts_dir = cfg["paths"]["artifacts"]
    ensure_dir(artifacts_dir)

    ds_name = cfg["dataset"]["name"]
    ds_cfg  = cfg["dataset"]["config"]

    corpus_path = f"{processed_dir}/{ds_name}_{ds_cfg}_corpus.jsonl"
    if not Path(corpus_path).exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}. Run build_corpus_from_queries.py first.")

    emb_cfg = cfg["embedding"]
    model_name = emb_cfg["model_name"]
    batch_size = int(emb_cfg.get("batch_size", 64))
    normalize = bool(emb_cfg.get("normalize", True))
    max_docs = emb_cfg.get("max_docs", None)

    # Load docs
    docs = []
    for doc in read_jsonl(corpus_path):
        docs.append(doc)
        if max_docs and len(docs) >= int(max_docs):
            break

    doc_ids = [d["doc_id"] for d in docs]
    texts = [d["text"] for d in docs]

    print(f"Loaded {len(docs)} docs from {corpus_path}")
    print("Embedding model:", model_name)

    # Embed
    model = SentenceTransformer(model_name)
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    embs = np.asarray(embs, dtype=np.float32)
    t1 = time.time()

    print(f"Embeddings shape: {embs.shape}, time: {(t1 - t0):.1f}s")

    # Build FAISS index
    dim = embs.shape[1]
    index_type = cfg["faiss"].get("index_type", "IndexFlatIP")

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    index.add(embs)
    print("FAISS index ntotal:", index.ntotal)

    # Save artifacts
    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}.index"
    ids_path   = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_doc_ids.json"
    meta_path  = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_meta.json"

    faiss.write_index(index, index_path)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    meta = {
        "dataset": f"{ds_name}:{ds_cfg}",
        "corpus_path": corpus_path,
        "embedding_model": model_name,
        "dim": int(dim),
        "normalize_embeddings": normalize,
        "index_type": index_type,
        "num_docs_indexed": int(len(docs)),
        "created_at_unix": time.time(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved index:", index_path)
    print("Saved ids:", ids_path)
    print("Saved meta:", meta_path)

if __name__ == "__main__":
    main()