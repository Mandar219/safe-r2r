import json
import time
from typing import Dict, Any, Iterator, List, Optional
from tqdm import tqdm

from safe_r2r.evaluation.metrics import exact_match
from safe_r2r.evaluation.token_overlap import precision_recall_f1
from safe_r2r.generation.postprocess import postprocess_answer
from safe_r2r.generation.prompting import build_rag_prompt

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


def run_rung(
    *,
    rung: int,
    queries_path: str,
    ladder,
    llm,
    log_path: str,
    metrics_path: str,
    llm_backend: str,
    llm_model: str,
    max_examples: Optional[int] = None,
    config_path: str = "",
) -> Dict[str, Any]:
    """
    Runs one rung evaluation using already-initialized ladder + llm.
    """
    ems, f1s = [], []
    precisions, recalls = [], []
    rt_ms, gt_ms = [], []
    prompt_toks, completion_toks, total_toks = [], [], []

    n = 0

    with open(log_path, "w", encoding="utf-8") as out_f:
        iterator = read_jsonl(queries_path)
        if max_examples is not None:
            total = int(max_examples)
        else:
            total = None  # tqdm will show indeterminate progress

        for ex in tqdm(
            iterator,
            total=total,
            desc=f"Rung {rung}",
            dynamic_ncols=True,
        ):
            if max_examples is not None and n >= int(max_examples):
                break

            qid = ex["qid"]
            q = ex["question"]
            gold = ex["answer"]

            # Retrieval
            t0 = time.perf_counter()
            docs = ladder.retrieve(q, rung=rung)
            t1 = time.perf_counter()

            # Prompt
            if rung == 0:
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

            # Metrics
            em = exact_match(pred, gold)
            prf = precision_recall_f1(pred, gold)

            # Latency
            retrieval_latency = (t1 - t0) * 1000.0
            meta = resp.get("meta", {}) or {}
            gen_latency = float(meta.get("latency_ms", (t3 - t2) * 1000.0))

            # Tokens
            pt = int(meta.get("prompt_tokens", -1))
            ct = int(meta.get("completion_tokens", -1))
            tt = int(meta.get("total_tokens", (pt + ct) if pt >= 0 and ct >= 0 else -1))

            # Uncertainty Signals
            faiss_scores = [float(d.score) for d in docs]  # note: for rung3/4 these might be rerank scores
            m = score_margin(faiss_scores)

            signals = {
                "is_insufficient": is_insufficient(pred),
                "is_yesno": is_yesno(pred),
                "ans_in_ctx_frac": answer_in_context_fraction(pred, docs),
                "score_top1": m["top1"],
                "score_margin12": m["margin12"],
                "score_margin1k": m["margin1k"],
            }

            # Log row
            row = {
                "qid": qid,
                "question": q,
                "gold_answer": gold,
                "raw_pred_answer": raw_pred,
                "pred_answer": pred,
                "rung": int(rung),
                "num_docs": len(docs),
                "retrieved": [{"doc_id": d.doc_id, "score": d.score, "title": d.title} for d in docs],
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": gen_latency,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "em": int(em),
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
                "llm_backend": llm_backend,
                "llm_model": llm_model,
            }
            row.update(signals)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Accumulate
            ems.append(em)
            precisions.append(prf.precision)
            recalls.append(prf.recall)
            f1s.append(prf.f1)
            rt_ms.append(retrieval_latency)
            gt_ms.append(gen_latency)
            if pt >= 0: prompt_toks.append(pt)
            if ct >= 0: completion_toks.append(ct)
            if tt >= 0: total_toks.append(tt)

            n += 1

    metrics = {
        "num_queries": int(n),
        "rung": int(rung),
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "em": _safe_mean(ems),
        "precision": _safe_mean(precisions),
        "recall": _safe_mean(recalls),
        "f1": _safe_mean(f1s),
        "avg_retrieval_ms": _safe_mean(rt_ms),
        "avg_generation_ms": _safe_mean(gt_ms),
        "avg_prompt_tokens": _safe_mean([float(x) for x in prompt_toks]),
        "avg_completion_tokens": _safe_mean([float(x) for x in completion_toks]),
        "avg_total_tokens": _safe_mean([float(x) for x in total_toks]),
        "log_path": log_path,
        "config_path": config_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics