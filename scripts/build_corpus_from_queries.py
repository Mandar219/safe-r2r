import json
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm

from safe_r2r.utils.io import load_yaml, ensure_dir, write_jsonl
from safe_r2r.utils.text import stable_doc_key, normalize_whitespace

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def hashed_doc_id(key: str) -> str:
    # stable across runs; extremely low collision risk with 16 hex
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"hp::{h}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--splits", type=str, default="train,validation")
    parser.add_argument("--out_scope", type=str, default="trainval")
    parser.add_argument(
        "--write_candidates",
        action="store_true",
        help="If set, writes per-split qid->candidate_doc_ids files (mostly for debugging).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    processed_dir = cfg["paths"]["data_processed"]
    ensure_dir(processed_dir)

    ds_name = cfg["dataset"]["name"]
    ds_cfg  = cfg["dataset"]["config"]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits provided")

    # Output files (combined corpus)
    corpus_path = f"{processed_dir}/{ds_name}_{ds_cfg}_{args.out_scope}_corpus.jsonl"
    lookup_path = f"{processed_dir}/{ds_name}_{ds_cfg}_{args.out_scope}_doc_lookup.json"

    # Dedup maps
    key_to_docid = {}
    docid_to_doc = {}
    corpus_rows = []

    # Optional: candidates per split
    candidate_rows_by_split = {sp: [] for sp in splits}

    # Process splits in fixed order for determinism
    for split in splits:
        queries_path = f"{processed_dir}/{ds_name}_{ds_cfg}_{split}_queries.jsonl"
        if not Path(queries_path).exists():
            raise FileNotFoundError(
                f"Missing queries jsonl: {queries_path}. Run make_queries_jsonl.py --split {split} first."
            )

        for ex in tqdm(read_jsonl(queries_path), desc=f"Building corpus from {split}"):
            qid = ex["qid"]
            candidates = []

            for para in ex["context"]:
                title = normalize_whitespace(para.get("title", ""))
                text  = normalize_whitespace(para.get("text", ""))

                if not title or not text:
                    continue

                key = stable_doc_key(title, text)
                if key not in key_to_docid:
                    doc_id = hashed_doc_id(key)

                    # In the extremely unlikely event of collision, extend hash
                    if doc_id in docid_to_doc and docid_to_doc[doc_id]["text"] != text:
                        h = hashlib.sha1((key + "::collision").encode("utf-8")).hexdigest()[:24]
                        doc_id = f"hp::{h}"

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

            if args.write_candidates:
                candidate_rows_by_split[split].append({
                    "qid": qid,
                    "candidate_doc_ids": candidates
                })

    # Write outputs
    write_jsonl(corpus_path, corpus_rows)
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(docid_to_doc, f, ensure_ascii=False)

    print("Wrote corpus:", corpus_path, "docs:", len(corpus_rows))
    print("Wrote lookup:", lookup_path)

    if args.write_candidates:
        for split in splits:
            cand_path = f"{processed_dir}/{ds_name}_{ds_cfg}_{split}_query_doc_candidates.jsonl"
            write_jsonl(cand_path, candidate_rows_by_split[split])
            print("Wrote candidates:", cand_path, "queries:", len(candidate_rows_by_split[split]))

if __name__ == "__main__":
    main()