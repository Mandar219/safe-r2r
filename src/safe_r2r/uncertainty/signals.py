import re
from typing import List, Optional, Dict

from safe_r2r.retrieval.faiss_retriever import RetrievedDoc


def is_insufficient(pred: str) -> int:
    return int(pred.strip().lower() == "insufficient evidence")


def is_yesno(pred: str) -> int:
    p = pred.strip().lower()
    return int(p == "yes" or p == "no")


def answer_in_context_fraction(pred: str, docs: List[RetrievedDoc]) -> float:
    """
    Cheap support proxy: fraction of answer tokens that appear anywhere in concatenated docs.
    Not perfect, but useful for gating.
    """
    p = pred.strip().lower()
    if p in ("", "insufficient evidence"):
        return 0.0

    # tokenize answer
    toks = re.findall(r"[a-z0-9]+", p)
    if not toks:
        return 0.0

    ctx = " ".join([d.text for d in docs]).lower()
    hit = sum(1 for t in toks if t in ctx)
    return float(hit / len(toks))


def score_margin(scores: List[float]) -> Dict[str, float]:
    """
    For faiss scores or rerank scores.
    Returns top1, top2, margin (top1-top2), and spread (top1-topK).
    """
    if not scores:
        return {"top1": 0.0, "top2": 0.0, "margin12": 0.0, "margin1k": 0.0}

    s = list(scores)
    top1 = float(s[0])
    top2 = float(s[1]) if len(s) > 1 else float(s[0])
    topk = float(s[-1])
    return {
        "top1": top1,
        "top2": top2,
        "margin12": float(top1 - top2),
        "margin1k": float(top1 - topk),
    }