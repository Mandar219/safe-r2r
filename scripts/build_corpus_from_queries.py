import json
from pathlib import Path
from tqdm import tqdm

from safe_r2r.utils.io import load_yaml, ensure_dir, write_jsonl
from safe_r2r.utils.text import stable_doc_key, normalize_whitespace

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    cfg = load_yaml("configs/default.yaml")
    processed_dir = cfg["paths"]["data_processed"]
    ensure_dir(processed_dir)

    # Input queries file
    ds_name = cfg["dataset"]["name"]
    ds_cfg  = cfg["dataset"]["config"]
    split   = cfg["dataset"]["split_valid"]
    queries_path = f"{processed_dir}/{ds_name}_{ds_cfg}_{split}_queries.jsonl"

    if not Path(queries_path).exists():
        raise FileNotFoundError(f"Missing queries jsonl: {queries_path}. Run make_queries_jsonl.py first.")

    # Output files
    corpus_path = f"{processed_dir}/{ds_name}_{ds_cfg}_corpus.jsonl"
    lookup_path = f"{processed_dir}/{ds_name}_{ds_cfg}_doc_lookup.json"
    cand_path   = f"{processed_dir}/{ds_name}_{ds_cfg}_{split}_query_doc_candidates.jsonl"

    # Dedup maps
    key_to_docid = {}
    docid_to_doc = {}

    # For stable incremental doc ids
    def make_doc_id(i: int) -> str:
        return f"hp::{i:08d}"

    corpus_rows = []
    candidate_rows = []

    next_id = 0

    for ex in tqdm(read_jsonl(queries_path), desc="Building corpus from queries"):
        qid = ex["qid"]
        candidates = []

        for para in ex["context"]:
            title = normalize_whitespace(para["title"])
            text  = normalize_whitespace(para["text"])

            if not title or not text:
                continue

            key = stable_doc_key(title, text)
            if key not in key_to_docid:
                doc_id = make_doc_id(next_id)
                next_id += 1
                key_to_docid[key] = doc_id

                doc_obj = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "source": f"{ds_name}:{ds_cfg}"
                }
                docid_to_doc[doc_id] = {"title": title, "text": text}
                corpus_rows.append(doc_obj)

            candidates.append(key_to_docid[key])

        # The distractor setting should give exactly 10
        candidate_rows.append({
            "qid": qid,
            "candidate_doc_ids": candidates
        })

    # Write outputs
    write_jsonl(corpus_path, corpus_rows)

    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(docid_to_doc, f, ensure_ascii=False)

    write_jsonl(cand_path, candidate_rows)

    print("Wrote corpus:", corpus_path, "docs:", len(corpus_rows))
    print("Wrote lookup:", lookup_path)
    print("Wrote candidates:", cand_path, "queries:", len(candidate_rows))

if __name__ == "__main__":
    main()