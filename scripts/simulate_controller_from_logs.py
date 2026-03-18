#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from safe_r2r.controller.scoring import compute_score
from safe_r2r.controller.policy import select_first_safe_rung


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rung_log(path: str) -> Dict[str, Dict[str, Any]]:
    return {row["qid"]: row for row in read_jsonl(path)}


def summarize(rows: List[Dict[str, Any]], rung_order: List[int]) -> Dict[str, Any]:
    certified = [r for r in rows if int(r.get("accepted", 0)) == 1]
    answered = [r for r in rows if int(r.get("answered", 0)) == 1]
    fallback_answered = [r for r in rows if int(r.get("fallback_used", 0)) == 1]
    unanswered = [r for r in rows if int(r.get("answered", 0)) == 0]

    summary: Dict[str, Any] = {
        "num_queries": len(rows),

        # certified subset
        "num_certified_accepted": len(certified),
        "certified_coverage": (len(certified) / len(rows)) if rows else 0.0,

        # all answered queries
        "num_answered_total": len(answered),
        "answer_coverage": (len(answered) / len(rows)) if rows else 0.0,

        # fallback / abstention distinction
        "num_fallback_answered": len(fallback_answered),
        "num_unanswered": len(unanswered),
        "abstention_rate": (len(unanswered) / len(rows)) if rows else 0.0,

        "accept_by_rung": {str(r): 0 for r in rung_order},
        "fallback_by_rung": {str(r): 0 for r in rung_order},
    }

    for r in certified:
        if r["selected_rung"] is not None:
            summary["accept_by_rung"][str(r["selected_rung"])] += 1

    for r in fallback_answered:
        if r["selected_rung"] is not None:
            summary["fallback_by_rung"][str(r["selected_rung"])] += 1

    # Metrics on certified subset
    if certified:
        em = np.array([int(r["em"]) for r in certified], dtype=int)
        summary["empirical_accuracy_certified"] = float(np.mean(em))
        summary["empirical_risk_certified"] = float(np.mean(1 - em))
    else:
        summary["empirical_accuracy_certified"] = None
        summary["empirical_risk_certified"] = None

    # Metrics on all answered queries (including fallback)
    if answered:
        em = np.array([int(r["em"]) for r in answered], dtype=int)
        summary["empirical_accuracy_answered"] = float(np.mean(em))
        summary["empirical_risk_answered"] = float(np.mean(1 - em))

        for key in [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "retrieval_latency_ms",
            "generation_latency_ms",
        ]:
            vals = [
                float(r[key]) for r in answered
                if key in r and r[key] is not None
            ]
            summary[f"avg_{key}_answered"] = float(np.mean(vals)) if vals else 0.0
    else:
        summary["empirical_accuracy_answered"] = None
        summary["empirical_risk_answered"] = None

    for rung in rung_order:
        sel = [r for r in certified if r["selected_rung"] == rung]
        summary[f"avg_score_rung{rung}"] = float(np.mean([float(r["score"]) for r in sel])) if sel else None

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config with rung logs, taus, weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary_out", required=True)
    ap.add_argument("--force_rung4_fallback", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    c = cfg["controller"]

    rung_logs_cfg = c["rung_logs"]
    taus_cfg = c["taus"]
    weights_cfg = c["weights"]
    knobs_cfg = c.get("feature_knobs", {})
    rung_order = [int(r) for r in c.get("rung_order", [0, 1, 2, 3, 4])]

    rung_logs = {
        int(r): load_rung_log(path)
        for r, path in rung_logs_cfg.items()
    }

    taus = {
        int(r): float(tau)
        for r, tau in taus_cfg.items()
    }

    # Intersect qids across all configured rung logs, including rung 0
    qid_sets = [set(v.keys()) for v in rung_logs.values()]
    common_qids = set.intersection(*qid_sets)
    qids = sorted(common_qids)

    def score_fn(row: Dict[str, Any]) -> float:
        return compute_score(
            row,
            wq=float(weights_cfg["wq"]),
            wv=float(weights_cfg["wv"]),
            ws=float(weights_cfg["ws"]),
            wi=float(weights_cfg["wi"]),
            wl=float(weights_cfg["wl"]),
            wt=float(weights_cfg["wt"]),
            topk_hit_k=int(knobs_cfg.get("top2_hit_k", 2)),
            length_limit=int(knobs_cfg.get("length_limit", 5)),
        )

    out_rows: List[Dict[str, Any]] = []

    for qid in qids:
        rows_by_rung = {r: rung_logs[r][qid] for r in rung_order}

        selected_rung, selected_score, selected_tau = select_first_safe_rung(
            rows_by_rung=rows_by_rung,
            taus=taus,
            score_fn=score_fn,
            rung_order=rung_order,
        )

        fallback_used = 0
        accepted = 1

        if selected_rung is None:
            if args.force_rung4_fallback:
                # final fallback should be the highest rung in the configured order
                selected_rung = max(rung_order)
                selected_score = score_fn(rows_by_rung[selected_rung])
                selected_tau = taus.get(selected_rung, None)
                fallback_used = 1
                accepted = 0
            else:
                out_rows.append({
                    "qid": qid,
                    "accepted": 0,
                    "answered": 0,
                    "selected_rung": None,
                    "fallback_used": 0,
                    "reason": "no_rung_passed",
                })
                continue

        chosen = rows_by_rung[selected_rung]

        out_rows.append({
            "qid": qid,
            "question": chosen.get("question"),
            "gold_answer": chosen.get("gold_answer"),
            "pred_answer": chosen.get("pred_answer"),
            "accepted": accepted,
            "answered": 1,  
            "selected_rung": selected_rung,
            "fallback_used": fallback_used,
            "score": selected_score,
            "tau": selected_tau,
            "em": int(chosen.get("em", 0)),
            "f1": float(chosen.get("f1", 0.0)),
            "qa_match_f1": float(chosen.get("qa_match_f1", 0.0)),
            "qa_match_em": int(chosen.get("qa_match_em", 0)),
            "verifier_sent_max_margin": float(chosen.get("verifier_sent_max_margin", 0.0)),
            "span_doc_hits": int(chosen.get("span_doc_hits", 0)),
            "prompt_tokens": chosen.get("prompt_tokens"),
            "completion_tokens": chosen.get("completion_tokens"),
            "total_tokens": chosen.get("total_tokens"),
            "retrieval_latency_ms": chosen.get("retrieval_latency_ms"),
            "generation_latency_ms": chosen.get("generation_latency_ms"),
        })

    write_jsonl(args.out, out_rows)
    summary = summarize(out_rows, rung_order=rung_order)

    summary["controller_config"] = c
    summary["force_rung4_fallback"] = bool(args.force_rung4_fallback)

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()