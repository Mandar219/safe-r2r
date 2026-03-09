#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from safe_r2r.utils.io import load_yaml, ensure_dir


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_hash_int(s: str, seed: str) -> int:
    h = hashlib.sha256((seed + "::" + s).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--calib_size", type=int, default=1000)
    parser.add_argument("--seed", type=str, default="safe_r2r")
    parser.add_argument(
        "--write_subset",
        action="store_true",
        help="Also write subset jsonl file for calibration.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]

    queries_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{args.split}_queries.jsonl'
    if not Path(queries_path).exists():
        raise FileNotFoundError(
            f"Missing queries file: {queries_path}\n"
            f"Create it first (generate {args.split} queries JSONL)."
        )

    splits_dir = f'{cfg["paths"]["artifacts"]}/splits'
    ensure_dir(splits_dir)

    rows = read_jsonl(queries_path)
    if not rows:
        raise ValueError(f"No rows found in {queries_path}")

    def get_qid(r: Dict[str, Any]) -> str:
        if "qid" in r:
            return str(r["qid"])
        if "id" in r:
            return str(r["id"])
        raise KeyError("Row has neither 'qid' nor 'id'")

    keyed: List[Tuple[int, Dict[str, Any]]] = []
    for r in rows:
        qid = get_qid(r)
        keyed.append((stable_hash_int(qid, args.seed), r))

    keyed.sort(key=lambda x: x[0])
    ordered = [r for _, r in keyed]

    if len(ordered) < args.calib_size:
        raise ValueError(
            f"Not enough {args.split} examples ({len(ordered)}) for calib_size={args.calib_size}. "
            f"Generate more {args.split} queries or reduce calib_size."
        )

    calib_rows = ordered[: args.calib_size]
    calib_qids = [get_qid(r) for r in calib_rows]

    calib_qids_path = f"{splits_dir}/calib_qids_{args.split}.json"
    with open(calib_qids_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "split": args.split, "qids": calib_qids}, f, indent=2)

    print("Saved calib qids:", calib_qids_path)
    print(f"Counts: calib={len(calib_qids)} from {queries_path}")

    if args.write_subset:
        base = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{args.split}'
        calib_out = f"{base}_calib.jsonl"
        write_jsonl(calib_out, calib_rows)
        print("Saved calib jsonl:", calib_out)


if __name__ == "__main__":
    main()