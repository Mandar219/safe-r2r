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
from safe_r2r.evaluation.metrics import exact_match
from safe_r2r.evaluation.token_overlap import precision_recall_f1
from safe_r2r.generation.prompting import build_rag_prompt
from safe_r2r.retrieval.faiss_retriever import FaissRetriever, RetrievedDoc
from safe_r2r.llm.base import LLMConfig
from safe_r2r.llm.factory import make_llm
from safe_r2r.generation.postprocess import postprocess_answer
from safe_r2r.uncertainty.signals import (
    is_insufficient,
    is_yesno,
    answer_in_context_fraction,
    score_margin,
)


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

    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

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
    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval.index"
    ids_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_doc_ids.json"
    meta_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_meta.json"
    doc_lookup_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_trainval_doc_lookup.json'

    for p in [index_path, ids_path, meta_path, doc_lookup_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Missing required artifact: {p}\n"
                f"Make sure indexing artifacts were built."
            )

    retriever = FaissRetriever(
        index_path=index_path,
        ids_path=ids_path,
        meta_path=meta_path,
        doc_lookup_path=doc_lookup_path,
    )

    llm_cfg = LLMConfig(**cfg["llm"])
    llm = make_llm(llm_cfg)

    top_k = args.top_k if args.top_k is not None else cfg.get("baseline", {}).get("top_k", 20)
    max_examples = args.max_examples if args.max_examples is not None else cfg["dataset"].get("max_valid_examples", None)

    tag = f"_{args.out_tag}" if args.out_tag else ""
    log_path = f'{cfg["paths"]["logs"]}/static_rag_top{top_k}{tag}.jsonl'
    metrics_path = f'{cfg["paths"]["artifacts"]}/static_rag_top{top_k}{tag}_metrics.json'

    ems: List[float] = []
    f1s: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    retrieval_ms: List[float] = []
    gen_ms: List[float] = []
    prompt_toks: List[int] = []
    completion_toks: List[int] = []
    total_toks: List[int] = []

    n = 0

    with open(log_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(read_jsonl(queries_path), desc=f"Static RAG top{top_k} ({llm_cfg.backend})"):
            if max_examples is not None and n >= int(max_examples):
                break

            qid = ex["qid"]
            question = ex["question"]
            gold = ex["answer"]

            # Retrieval
            t0 = time.perf_counter()
            docs: List[RetrievedDoc] = retriever.search(question, top_k=top_k)
            t1 = time.perf_counter()
            rt_ms = (t1 - t0) * 1000.0

            # Prompt
            prompt = build_rag_prompt(question, docs)

            # Generation
            t2 = time.perf_counter()
            resp = llm.generate(prompt)
            t3 = time.perf_counter()

            raw_pred = (resp.get("text") or "").strip()
            pred = postprocess_answer(raw_pred)

            meta = resp.get("meta", {}) or {}
            gt_ms = float(meta.get("latency_ms", (t3 - t2) * 1000.0))

            pt = int(meta.get("prompt_tokens", -1))
            ct = int(meta.get("completion_tokens", -1))
            tt = int(meta.get("total_tokens", (pt + ct) if pt >= 0 and ct >= 0 else -1))

            # Metrics
            em = exact_match(pred, gold)
            prf = precision_recall_f1(pred, gold)

            # Uncertainty signals
            faiss_scores = [float(d.score) for d in docs]
            m = score_margin(faiss_scores)

            signals = {
                "is_insufficient": is_insufficient(pred),
                "is_yesno": is_yesno(pred),
                "ans_in_ctx_frac": answer_in_context_fraction(pred, docs),
                "score_top1": m["top1"],
                "score_margin12": m["margin12"],
                "score_margin1k": m["margin1k"],
            }

            retrieved_logged = []
            for d in docs:
                text = getattr(d, "text", "") or ""
                retrieved_logged.append({
                    "doc_id": d.doc_id,
                    "score": float(d.score),
                    "title": d.title,
                    "text": text,
                    "text_chars": len(text) if isinstance(text, str) else 0,
                })

            row = {
                "qid": qid,
                "question": question,
                "gold_answer": gold,
                "raw_pred_answer": raw_pred,
                "pred_answer": pred,
                "rung": -1,  # static baseline
                "top_k": int(top_k),
                "num_docs": len(docs),
                "retrieved": retrieved_logged,
                "retrieval_latency_ms": rt_ms,
                "generation_latency_ms": gt_ms,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "em": int(em),
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
                "llm_backend": llm_cfg.backend,
                "llm_model": llm_cfg.model_name,
            }
            row.update(signals)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            ems.append(em)
            precisions.append(prf.precision)
            recalls.append(prf.recall)
            f1s.append(prf.f1)
            retrieval_ms.append(rt_ms)
            gen_ms.append(gt_ms)

            if pt >= 0:
                prompt_toks.append(pt)
            if ct >= 0:
                completion_toks.append(ct)
            if tt >= 0:
                total_toks.append(tt)

            n += 1

    metrics = {
        "dataset": f"{ds_name}:{ds_cfg}:{split}",
        "num_queries": int(n),
        "rung": -1,
        "top_k": int(top_k),
        "llm_backend": llm_cfg.backend,
        "llm_model": llm_cfg.model_name,
        "em": _safe_mean(ems),
        "precision": _safe_mean(precisions),
        "recall": _safe_mean(recalls),
        "f1": _safe_mean(f1s),
        "avg_prompt_tokens": _safe_mean([float(x) for x in prompt_toks]),
        "avg_completion_tokens": _safe_mean([float(x) for x in completion_toks]),
        "avg_total_tokens": _safe_mean([float(x) for x in total_toks]),
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