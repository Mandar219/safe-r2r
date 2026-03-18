import re
from typing import Any, Dict


def clip01(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pred_len_tokens(row: Dict[str, Any]) -> int:
    ans = str(row.get("pred_answer", "") or "").strip().lower()
    return len(re.findall(r"[A-Za-z0-9]+", ans))


def is_yesno_answer(row: Dict[str, Any]) -> bool:
    ans = str(row.get("pred_answer", "") or "").strip().lower()
    return ans in {"yes", "no"}


def is_date_like_answer(row: Dict[str, Any]) -> bool:
    ans = str(row.get("pred_answer", "") or "").strip().lower()
    months = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    }
    if any(m in ans for m in months):
        return True
    if re.search(r"\b\d{4}\b", ans):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", ans):
        return True
    return False


def answer_in_topk(row: Dict[str, Any], topk: int = 2) -> float:
    ans = normalize_text(row.get("pred_answer", ""))
    if not ans or ans in {"insufficient evidence", "unknown", "none", "not enough information"}:
        return 0.0

    docs = row.get("retrieved", [])[:topk]
    for d in docs:
        txt = normalize_text(
            d.get("text", "") or d.get("contents", "") or d.get("content", "") or ""
        )
        if ans and ans in txt:
            return 1.0
    return 0.0


def span_support_from_hits(row: Dict[str, Any]) -> float:
    try:
        hits = float(row.get("span_doc_hits", 0.0))
    except Exception:
        hits = 0.0
    return 1.0 if hits >= 2.0 else 0.0


def too_long_flag(row: Dict[str, Any], token_limit: int = 5) -> float:
    if is_yesno_answer(row) or is_date_like_answer(row):
        return 0.0
    return 1.0 if pred_len_tokens(row) > token_limit else 0.0


def insufficient_flag(row: Dict[str, Any]) -> float:
    try:
        if int(row.get("is_insufficient", 0)) != 0:
            return 1.0
    except Exception:
        pass

    ans = normalize_text(row.get("pred_answer", ""))
    return 1.0 if ans == "insufficient evidence" else 0.0


def compute_score(
    row: Dict[str, Any],
    wq: float = 6.0,
    wv: float = 2.0,
    ws: float = 2.0,
    wi: float = 4.0,
    wl: float = 2.0,
    wt: float = 2.0,
    qa_key: str = "qa_match_f1",
    verifier_key: str = "verifier_sent_max_margin",
    topk_hit_k: int = 2,
    length_limit: int = 5,
) -> float:
    q = clip01(row.get(qa_key, 0.0))
    v = clip01(row.get(verifier_key, 0.0))
    s = span_support_from_hits(row)
    i = insufficient_flag(row)
    l = too_long_flag(row, token_limit=length_limit)
    t = answer_in_topk(row, topk=topk_hit_k)

    return (
        wq * (1.0 - q) +
        wv * (1.0 - v) +
        ws * (1.0 - s) +
        wi * i +
        wl * l +
        wt * (1.0 - t)
    )