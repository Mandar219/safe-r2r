import re
import string
from collections import Counter
from dataclasses import dataclass

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text

@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    common: int
    pred_len: int
    gold_len: int

def precision_recall_f1(pred: str, gold: str) -> PRF:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()

    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return PRF(1.0, 1.0, 1.0, 0, 0, 0)
    if len(pred_toks) == 0:
        return PRF(0.0, 0.0, 0.0, 0, 0, len(gold_toks))
    if len(gold_toks) == 0:
        return PRF(0.0, 0.0, 0.0, 0, len(pred_toks), 0)

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return PRF(0.0, 0.0, 0.0, 0, len(pred_toks), len(gold_toks))

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return PRF(precision, recall, f1, num_same, len(pred_toks), len(gold_toks))