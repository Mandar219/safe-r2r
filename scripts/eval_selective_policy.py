#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def read_thresholds(path: str):
    obj = json.load(open(path, "r", encoding="utf-8"))
    return obj["thresholds"]

def compute_S(row, eps, ws, wm, wi):
    support = float(row.get("ans_in_ctx_frac", 0.0))
    margin = float(row.get("score_margin12", 0.0))
    margin = max(margin, 0.0)
    insuff = int(row.get("is_insufficient", 0))
    return ws * (1.0 - support) + wm * (1.0 / (eps + margin)) + wi * float(insuff)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_log", type=str, required=True)
    ap.add_argument("--thresholds", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    eval_rows = read_jsonl(args.eval_log)
    thrs = read_thresholds(args.thresholds)

    Path(Path(args.out_path).parent).mkdir(parents=True, exist_ok=True)

    results = {"eval_log": args.eval_log, "thresholds": args.thresholds, "results": []}

    for t in thrs:
        alpha = t["alpha"]
        tau = t["tau"]
        ws, wm, wi, eps = t["ws"], t["wm"], t["wi"], t["eps"]

        accept = []
        for r in eval_rows:
            S = compute_S(r, eps, ws, wm, wi)
            if S <= tau:
                accept.append(r)

        n = len(eval_rows)
        na = len(accept)

        if na > 0:
            em = sum(int(r.get("em", 0)) for r in accept) / na
            f1 = sum(float(r.get("f1", 0.0)) for r in accept) / na
            risk = sum(1 - int(r.get("em", 0)) for r in accept) / na
            avg_tokens = sum(float(r.get("total_tokens", 0.0)) for r in accept) / na
        else:
            em = f1 = risk = avg_tokens = 0.0

        results["results"].append({
            "alpha": alpha,
            "tau": tau,
            "accept_rate": na / n,
            "n_eval": n,
            "n_accept": na,
            "em_on_accepted": em,
            "f1_on_accepted": f1,
            "risk_on_accepted": risk,
            "avg_total_tokens_on_accepted": avg_tokens,
        })

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Wrote:", args.out_path)
    for r in results["results"]:
        print(
            f'alpha={r["alpha"]} accept={r["accept_rate"]:.3f} '
            f'risk={r["risk_on_accepted"]:.3f} em={r["em_on_accepted"]:.3f} '
            f'f1={r["f1_on_accepted"]:.3f}'
        )

if __name__ == "__main__":
    main()