#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re
import numpy as np
from scipy.stats import beta


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clip01(x):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def compute_score(
    row: Dict[str, Any],
    wq: float,
    wv: float,
    ws: float,
    wi: float,
    wl: float = 2.0,
    wt: float = 2.0,
    qa_f1_key: str = "qa_match_f1",
    ver_key: str = "verifier_sent_max_margin",
    span_docs_key: str = "span_doc_hits",
    insuff_key: str = "is_insufficient",
) -> float:
    """
    Lower score = safer.
    """
    qa_f1 = clip01(row.get(qa_f1_key, 0.0))
    ver = clip01(row.get(ver_key, 0.0))
    span_docs = safe_float(row.get(span_docs_key, 0.0), 0.0)
    insuff = 1.0 if int(row.get(insuff_key, 0)) != 0 else 0.0

    # Stronger multi-hop-ish support: 1 if answer appears in >=2 docs else 0
    span_support = 1.0 if span_docs >= 2.0 else 0.0

    plen = pred_len_tokens(row)
    too_long = 0.0
    if not is_yesno_answer(row) and not is_date_like_answer(row):
        too_long = 1.0 if plen > 5 else 0.0

    top2_hit = answer_in_top2(row)

    return (
        wq * (1.0 - qa_f1) +
        wv * (1.0 - ver) +
        ws * (1.0 - span_support) +
        wi * insuff +
        wl * too_long +
        wt * (1.0 - top2_hit)
    )

def pred_len_tokens(row):
    ans = str(row.get("pred_answer", "") or "").strip().lower()
    toks = re.findall(r"[A-Za-z0-9]+", ans)
    return len(toks)

def is_yesno_answer(row):
    ans = str(row.get("pred_answer", "") or "").strip().lower()
    return ans in {"yes", "no"}

def is_date_like_answer(row):
    ans = str(row.get("pred_answer", "") or "").strip().lower()
    months = {
        "january","february","march","april","may","june",
        "july","august","september","october","november","december"
    }
    if any(m in ans for m in months):
        return True
    if re.search(r"\b\d{4}\b", ans):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", ans):
        return True
    return False

def normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def answer_in_top2(row):
    ans = normalize_text(row.get("pred_answer", ""))
    if not ans or ans in {"insufficient evidence", "unknown", "none", "not enough information"}:
        return 0.0
    docs = row.get("retrieved", [])[:2]
    for d in docs:
        txt = normalize_text(d.get("text", "") or d.get("contents", "") or d.get("content", "") or "")
        if ans and ans in txt:
            return 1.0
    return 0.0

def cp_upper_bound(num_errors: int, n: int, delta: float) -> float:
    """
    One-sided Clopper-Pearson upper confidence bound for Bernoulli mean.
    Confidence level = 1 - delta.
    """
    if n == 0:
        return 1.0
    if num_errors == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, num_errors + 1, n - num_errors))


def select_tau_cp(
    rows: List[Dict[str, Any]],
    scores: np.ndarray,
    alpha: float,
    delta: float,
    k_min: int,
    label_key: str = "em",
) -> Tuple[Optional[float], int, float, float]:
    order = np.argsort(scores)  # lowest = safest
    sorted_scores = scores[order]
    sorted_err = np.array([1 - int(rows[i].get(label_key, 0)) for i in order], dtype=int)

    best_tau = None
    best_k = 0
    best_ucb = 1.0
    best_emp = 1.0

    cum_err = np.cumsum(sorted_err)

    for k in range(k_min, len(sorted_scores) + 1):
        err_k = int(cum_err[k - 1])
        emp = err_k / k
        ucb = cp_upper_bound(err_k, k, delta)
        if ucb <= alpha:
            best_tau = float(sorted_scores[k - 1])
            best_k = k
            best_ucb = ucb
            best_emp = emp

    return best_tau, best_k, best_ucb, best_emp


def bucket_risk(rows: List[Dict[str, Any]], scores: np.ndarray, frac: float, label_key: str = "em"):
    n = len(scores)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(scores)[:k]
    y = np.array([int(rows[int(i)].get(label_key, 0)) for i in idx], dtype=int)
    risk = float(np.mean(1 - y))
    return k, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--alphas", default="0.2,0.1,0.05")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k_min", type=int, default=50)
    ap.add_argument("--wq", type=float, default=4.0)
    ap.add_argument("--wv", type=float, default=2.0)
    ap.add_argument("--ws", type=float, default=2.0)
    ap.add_argument("--wi", type=float, default=4.0)
    ap.add_argument("--wl", type=float, default=2.0)
    ap.add_argument("--wt", type=float, default=2.0)
    ap.add_argument("--label_key", default="em")
    args = ap.parse_args()

    rows = list(read_jsonl(args.log))
    print("Rows:", len(rows))

    scores = np.array([
        compute_score(
            r,
            wq=args.wq,
            wv=args.wv,
            ws=args.ws,
            wi=args.wi,
        )
        for r in rows
    ], dtype=float)

    print(f"weights: wq={args.wq} wv={args.wv} ws={args.ws} wi={args.wi}")
    print("---- tail bucket risks (lower is better) ----")
    for frac in [0.01, 0.02, 0.05, 0.10, 0.20]:
        k, risk = bucket_risk(rows, scores, frac, label_key=args.label_key)
        print(f"top {int(frac*100)}% (k={k}): emp_risk={risk:.3f}  (emp_acc={1-risk:.3f})")

    print("---- CP-certified thresholds ----")
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    for alpha in alphas:
        tau, k, ucb, emp = select_tau_cp(
            rows=rows,
            scores=scores,
            alpha=alpha,
            delta=args.delta,
            k_min=args.k_min,
            label_key=args.label_key,
        )
        if tau is None:
            print(f"alpha={alpha:.3f}: no feasible tau with k_min={args.k_min}")
        else:
            print(
                f"alpha={alpha:.3f}: tau={tau:.6f} accept_k={k} "
                f"emp_risk={emp:.3f} cp_ucb={ucb:.3f} emp_acc={1-emp:.3f}"
            )


if __name__ == "__main__":
    main()