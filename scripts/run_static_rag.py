#!/usr/bin/env python3
"""
Static RAG baseline runner (local LLM backend).

Pipeline:
  - load queries jsonl
  - load FAISS index + doc lookup
  - retrieve top_k docs
  - build RAG prompt
  - generate answer using local HF LLM (or Mock) via safe_r2r.llm.factory
  - compute EM/F1
  - write per-query logs to logs/static_rag_top{K}.jsonl
  - write aggregate metrics to artifacts/static_rag_top{K}_metrics.json

Expected config keys in configs/default.yaml:
paths:
  data_processed: data/processed
  logs: logs
  artifacts: artifacts

dataset:
  name: hotpot_qa
  config: distractor
  split_valid: validation
  max_valid_examples: 50   # optional cap

baseline:
  top_k: 20

llm:
  backend: mock | hf_local
  model_name: Qwen/Qwen2.5-7B-Instruct
  temperature: 0.2
  max_new_tokens: 128
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterator, List

from tqdm import tqdm

from safe_r2r.utils.io import load_yaml, ensure_dir
from safe_r2r.evaluation.metrics import exact_match, f1_score
from safe_r2r.generation.prompting import build_rag_prompt
from safe_r2r.retrieval.faiss_retriever import FaissRetriever, RetrievedDoc
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
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--top_k", type=int, default=None, help="Override baseline.top_k")
    parser.add_argument("--max_examples", type=int, default=None, help="Override dataset.max_valid_examples")
    parser.add_argument("--out_tag", type=str, default="", help="Optional tag appended to output filenames")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Ensure output dirs exist
    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

    # Dataset paths
    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]
    split = cfg["dataset"]["split_valid"]

    queries_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{split}_queries.jsonl'
    if not Path(queries_path).exists():
        raise FileNotFoundError(
            f"Missing queries file: {queries_path}\n"
            f"Run: python scripts/make_queries_jsonl.py"
        )

    artifacts_dir = cfg["paths"]["artifacts"]
    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}.index"
    ids_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_doc_ids.json"
    meta_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_meta.json"

    doc_lookup_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_doc_lookup.json'

    for p in [index_path, ids_path, meta_path, doc_lookup_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Missing required artifact: {p}\n"
                f"Make sure you completed Day 3 and Day 4."
            )

    # Retriever
    retriever = FaissRetriever(
        index_path=index_path,
        ids_path=ids_path,
        meta_path=meta_path,
        doc_lookup_path=doc_lookup_path,
    )

    # LLM backend
    llm_cfg_dict = cfg["llm"].copy()
    llm_cfg = LLMConfig(**llm_cfg_dict)
    llm = make_llm(llm_cfg)

    # Run params
    top_k = args.top_k if args.top_k is not None else cfg.get("baseline", {}).get("top_k", 20)
    max_examples = args.max_examples if args.max_examples is not None else cfg["dataset"].get("max_valid_examples", None)

    tag = f"_{args.out_tag}" if args.out_tag else ""
    log_path = f'{cfg["paths"]["logs"]}/static_rag_top{top_k}{tag}.jsonl'
    metrics_path = f'{cfg["paths"]["artifacts"]}/static_rag_top{top_k}{tag}_metrics.json'

    # Metrics accumulators
    ems: List[float] = []
    f1s: List[float] = []
    retrieval_ms: List[float] = []
    gen_ms: List[float] = []

    n = 0

    # Stream writing to avoid keeping everything in memory
    with open(log_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(read_jsonl(queries_path), desc=f"Static RAG top{top_k} ({llm_cfg.backend})"):
            if max_examples is not None and n >= int(max_examples):
                break

            qid = ex["qid"]
            question = ex["question"]
            gold = ex["answer"]

            # --- Retrieval ---
            t0 = time.perf_counter()
            docs: List[RetrievedDoc] = retriever.search(question, top_k=top_k)
            t1 = time.perf_counter()
            rt_ms = (t1 - t0) * 1000.0

            # --- Prompt ---
            prompt = build_rag_prompt(question, docs)

            # --- Generation ---
            t2 = time.perf_counter()
            resp = llm.generate(prompt)
            t3 = time.perf_counter()

            pred = (resp.get("text") or "").strip()
            # Prefer model-reported latency if present; fallback to measured
            gt_ms = float(resp.get("meta", {}).get("latency_ms", (t3 - t2) * 1000.0))

            # --- Metrics ---
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)

            # Log row
            row = {
                "qid": qid,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "top_k": int(top_k),
                "retrieved": [
                    {"doc_id": d.doc_id, "score": d.score, "title": d.title}
                    for d in docs
                ],
                "retrieval_latency_ms": rt_ms,
                "generation_latency_ms": gt_ms,
                "llm_backend": llm_cfg.backend,
                "llm_model": llm_cfg.model_name,
                "em": int(em),
                "f1": float(f1),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Accumulate
            ems.append(em)
            f1s.append(f1)
            retrieval_ms.append(rt_ms)
            gen_ms.append(gt_ms)
            n += 1

    metrics = {
        "dataset": f"{ds_name}:{ds_cfg}:{split}",
        "num_queries": int(n),
        "top_k": int(top_k),
        "llm_backend": llm_cfg.backend,
        "llm_model": llm_cfg.model_name,
        "em": _safe_mean(ems),
        "f1": _safe_mean(f1s),
        "avg_retrieval_ms": _safe_mean(retrieval_ms),
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