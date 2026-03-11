#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterator, List

from tqdm import tqdm

from safe_r2r.utils.io import load_yaml, ensure_dir
from safe_r2r.llm.base import LLMConfig
from safe_r2r.llm.factory import make_llm
from safe_r2r.generation.prompting import build_rag_prompt
from safe_r2r.generation.postprocess import postprocess_answer
from safe_r2r.evaluation.metrics import exact_match, f1_score
from safe_r2r.evaluation.token_overlap import precision_recall_f1

from safe_r2r.retrieval.faiss_retriever import FaissRetriever
from safe_r2r.retrieval.ladder import RetrievalLadder, LadderConfig
from safe_r2r.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from safe_r2r.retrieval.compress import ExtractiveCompressor, CompressionConfig


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
    parser.add_argument("--rung", type=int, required=True, help="0=no retrieval, 1/2/3 = FAISS top-k per config")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--out_tag", type=str, default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]
    split = cfg["dataset"]["split_valid"]
    queries_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{split}_queries.jsonl'
    if not Path(queries_path).exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    # Artifacts
    artifacts_dir = cfg["paths"]["artifacts"]
    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval.index"
    ids_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_doc_ids.json"
    meta_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_meta.json"
    doc_lookup_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_trainval_doc_lookup.json'

    # Retriever (only used for rung > 0)
    retriever = FaissRetriever(index_path, ids_path, meta_path, doc_lookup_path)

    # Ladder config parsing
    rung_top_k_raw = cfg.get("ladder", {}).get("rung_top_k", {})
    rung_top_k = {int(k): int(v) for k, v in rung_top_k_raw.items()}

    rer_cfg = cfg.get("reranker", {})
    rer_enabled = bool(rer_cfg.get("enabled", False))
    reranker = None
    rerank_faiss_top_n = int(rer_cfg.get("faiss_top_n", 50))

    if rer_enabled:
        reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name=rer_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                batch_size=int(rer_cfg.get("batch_size", 32)),
            )
        )

    comp_cfg = cfg.get("compression", {})
    comp_enabled = bool(comp_cfg.get("enabled", False))
    compressor = None

    if comp_enabled:
        compressor = ExtractiveCompressor(
            CompressionConfig(
                embedder_model_name=comp_cfg.get("embedder_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                batch_size=int(comp_cfg.get("batch_size", 64)),
                max_sents_per_doc=int(comp_cfg.get("max_sents_per_doc", 4)),
                max_words_per_sent=int(comp_cfg.get("max_words_per_sent", 35)),
                max_words_per_doc=int(comp_cfg.get("max_words_per_doc", 140)),
            )
        )

    ladder = RetrievalLadder(
        retriever,
        LadderConfig(rung_top_k=rung_top_k, rerank_faiss_top_n=rerank_faiss_top_n),
        reranker=reranker,
        compressor=compressor,
    )

    # LLM
    llm_cfg = LLMConfig(**cfg["llm"])
    llm = make_llm(llm_cfg)

    tag = f"_{args.out_tag}" if args.out_tag else ""
    log_path = f'{cfg["paths"]["logs"]}/rung{args.rung}{tag}.jsonl'
    metrics_path = f'{cfg["paths"]["artifacts"]}/rung{args.rung}{tag}_metrics.json'

    ems, f1s = [], []
    rt_ms, gt_ms = [], []
    precisions, recalls = [], []
    prompt_toks, completion_toks, total_toks = [], [], []
    n = 0

    with open(log_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(read_jsonl(queries_path), desc=f"Rung {args.rung}"):
            if args.max_examples is not None and n >= int(args.max_examples):
                break

            qid = ex["qid"]
            q = ex["question"]
            gold = ex["answer"]

            # Retrieval
            t0 = time.perf_counter()
            docs = ladder.retrieve(q, rung=args.rung)
            t1 = time.perf_counter()

            # Prompt
            if args.rung == 0:
                # parametric prompt (no docs)
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
                    f"Question: {q}\n"
                    "Final answer:"
                )
            else:
                prompt = build_rag_prompt(q, docs)

            # Generation
            t2 = time.perf_counter()
            resp = llm.generate(prompt)
            t3 = time.perf_counter()

            raw_pred = (resp.get("text") or "").strip()
            pred = postprocess_answer(raw_pred)

            em = exact_match(pred, gold)
            # f1 = f1_score(pred, gold)
            prf = precision_recall_f1(pred, gold)  # token overlap P/R/F1
            f1 = prf.f1

            retrieval_latency = (t1 - t0) * 1000.0
            meta = resp.get("meta", {}) or {}
            gen_latency = float(meta.get("latency_ms", (t3 - t2) * 1000.0))

            pt = int(meta.get("prompt_tokens", -1))
            ct = int(meta.get("completion_tokens", -1))
            tt = int(meta.get("total_tokens", (pt + ct) if pt >= 0 and ct >= 0 else -1))

            row = {
                "qid": qid,
                "question": q,
                "gold_answer": gold,
                "raw_pred_answer": raw_pred,
                "pred_answer": pred,
                "rung": int(args.rung),
                "num_docs": len(docs),
                "retrieved": [{"doc_id": d.doc_id, "score": d.score, "title": d.title} for d in docs],
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": gen_latency,
                "em": int(em),
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
                "common_tokens": prf.common,
                "pred_tokens": prf.pred_len,
                "gold_tokens": prf.gold_len,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "llm_backend": llm_cfg.backend,
                "llm_model": llm_cfg.model_name,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            ems.append(em); f1s.append(f1)
            rt_ms.append(retrieval_latency); gt_ms.append(gen_latency)
            precisions.append(prf.precision)
            recalls.append(prf.recall)

            if pt >= 0: prompt_toks.append(pt)
            if ct >= 0: completion_toks.append(ct)
            if tt >= 0: total_toks.append(tt)
            n += 1

    metrics = {
        "dataset": f"{ds_name}:{ds_cfg}:{split}",
        "num_queries": int(n),
        "rung": int(args.rung),
        "top_k": 0 if args.rung == 0 else ladder.top_k_for_rung(args.rung),
        "llm_backend": llm_cfg.backend,
        "llm_model": llm_cfg.model_name,
        "em": _safe_mean(ems),
        "f1": _safe_mean(f1s),
        "precision": _safe_mean(precisions),
        "recall": _safe_mean(recalls),
        "avg_prompt_tokens": _safe_mean([float(x) for x in prompt_toks]),
        "avg_completion_tokens": _safe_mean([float(x) for x in completion_toks]),
        "avg_total_tokens": _safe_mean([float(x) for x in total_toks]),
        "avg_retrieval_ms": _safe_mean(rt_ms),
        "avg_generation_ms": _safe_mean(gt_ms),
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