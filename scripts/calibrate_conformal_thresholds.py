#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def compute_S(row: Dict[str, Any], eps: float, ws: float, wm: float, wi: float) -> float:
    support = float(row.get("ans_in_ctx_frac", 0.0))
    margin = float(row.get("score_margin12", 0.0))
    margin = max(margin, 0.0)
    insuff = int(row.get("is_insufficient", 0))

    return ws * (1.0 - support) + wm * (1.0 / (eps + margin)) + wi * float(insuff)

def calibrate_threshold(rows: List[Dict[str, Any]], alpha: float, eps: float, ws: float, wm: float, wi: float):
    scored = []
    for r in rows:
        S = compute_S(r, eps=eps, ws=ws, wm=wm, wi=wi)
        err = 0 if int(r.get("em", 0)) == 1 else 1
        scored.append((S, err))

    scored.sort(key=lambda x: x[0])

    # prefix scan
    errs = 0
    best_k = 0
    best_tau = None

    for k, (S, err) in enumerate(scored, start=1):
        errs += err
        risk = errs / k
        if risk <= alpha:
            best_k = k
            best_tau = S

    # If even the lowest-risk example violates, accept none
    if best_k == 0:
        best_tau = float("-inf")

    accept_rate = best_k / len(scored)
    # risk on accepted (calibration)
    cal_risk = (sum(e for _, e in scored[:best_k]) / best_k) if best_k > 0 else 0.0

    return {
        "alpha": alpha,
        "tau": best_tau,
        "accept_rate_calib": accept_rate,
        "risk_calib": cal_risk,
        "n_calib": len(scored),
        "n_accept": best_k,
        "ws": ws,
        "wm": wm,
        "wi": wi,
        "eps": eps,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_log", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--alphas", type=str, default="0.2,0.1,0.05")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--ws", type=float, default=1.0)
    ap.add_argument("--wm", type=float, default=0.2)
    ap.add_argument("--wi", type=float, default=0.5)
    args = ap.parse_args()

    rows = read_jsonl(args.calib_log)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    ensure_dir(str(Path(args.out_path).parent))

    out = {
        "calib_log": args.calib_log,
        "thresholds": [],
    }

    for a in alphas:
        out["thresholds"].append(calibrate_threshold(rows, a, args.eps, args.ws, args.wm, args.wi))

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out_path)
    for t in out["thresholds"]:
        print(
            f'alpha={t["alpha"]} tau={t["tau"]:.4f} '
            f'accept_calib={t["accept_rate_calib"]:.3f} risk_calib={t["risk_calib"]:.3f}'
        )

if __name__ == "__main__":
    main()