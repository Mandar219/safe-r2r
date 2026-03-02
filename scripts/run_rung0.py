#!/usr/bin/env python3
"""
Run Rung 0 (No retrieval) baseline.

- Loads queries JSONL
- Uses LLM backend via safe_r2r.llm.factory (mock or hf_local)
- Postprocesses answers with safe_r2r.generation.postprocess.postprocess_answer
- Computes EM / F1 using safe_r2r.evaluation.metrics
- Writes per-query logs and aggregate metrics

Usage:
  python scripts/run_rung0.py --config configs/default.yaml --max_examples 50 --out_tag rung0_test
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterator, List

from tqdm import tqdm

from safe_r2r.utils.io import load_yaml, ensure_dir
from safe_r2r.evaluation.metrics import exact_match, f1_score
from safe_r2r.generation.postprocess import postprocess_answer
from safe_r2r.llm.base import LLMConfig
from safe_r2r.llm.factory import make_llm


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--out_tag", type=str, default="rung0")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]
    split = cfg["dataset"]["split_valid"]

    queries_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{split}_queries.jsonl'
    if not Path(queries_path).exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}.")

    # LLM setup
    llm_cfg_dict = cfg["llm"].copy()
    llm_cfg = LLMConfig(**llm_cfg_dict)
    llm = make_llm(llm_cfg)

    tag = f"_{args.out_tag}" if args.out_tag else ""
    log_path = f'{cfg["paths"]["logs"]}/rung0{tag}.jsonl'
    metrics_path = f'{cfg["paths"]["artifacts"]}/rung0{tag}_metrics.json'

    ems: List[float] = []
    f1s: List[float] = []
    gen_ms: List[float] = []

    n = 0

    with open(log_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(read_jsonl(queries_path), desc="Rung0 (no retrieval)"):
            if args.max_examples is not None and n >= int(args.max_examples):
                break

            qid = ex["qid"]
            question = ex["question"]
            gold = ex["answer"]

            # Build a short instruction-only prompt for rung 0
            prompt = (
                "Answer the question concisely. Use only your internal knowledge.\n\n"
                "Output Rules (MUST FOLLOW):\n"
                "1) Output ONLY the final answer as a short span (one line).\n"
                "2) Do NOT explain.\n"
                "3) Do NOT add extra words.\n"
                "4) Do NOT include quotes.\n"
                "5) Do NOT repeat the question.\n"
                "6) If yes/no, output exactly: yes OR no\n"
                "7) If the documents do not contain enough information, output exactly: Insufficient evidence\n\n"
                f"Question: {question}\n"
                "Final answer:"
            )

            t0 = time.perf_counter()
            resp = llm.generate(prompt)
            t1 = time.perf_counter()

            raw_pred = (resp.get("text") or "").strip()
            gen_lat_ms = float(resp.get("meta", {}).get("latency_ms", (t1 - t0) * 1000.0))

            pred = postprocess_answer(raw_pred)

            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)

            row = {
                "qid": qid,
                "question": question,
                "gold_answer": gold,
                "raw_pred_answer": raw_pred,
                "pred_answer": pred,
                "generation_latency_ms": gen_lat_ms,
                "em": int(em),
                "f1": float(f1),
                "llm_backend": llm_cfg.backend,
                "llm_model": llm_cfg.model_name,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            ems.append(em)
            f1s.append(f1)
            gen_ms.append(gen_lat_ms)
            n += 1

    metrics = {
        "dataset": f"{ds_name}:{ds_cfg}:{split}",
        "num_queries": int(n),
        "llm_backend": llm_cfg.backend,
        "llm_model": llm_cfg.model_name,
        "em": _safe_mean(ems),
        "f1": _safe_mean(f1s),
        "avg_generation_ms": _safe_mean(gen_ms),
        "log_path": log_path,
        "config_path": args.config,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved log:", log_path)
    print("Saved metrics:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()