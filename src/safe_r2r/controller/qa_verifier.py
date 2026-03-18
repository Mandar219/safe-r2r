import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1

    overlap = 0
    used = {}
    for t in g:
        if common.get(t, 0) > used.get(t, 0):
            overlap += 1
            used[t] = used.get(t, 0) + 1

    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def pick_pred_answer(row: Dict[str, Any]) -> str:
    for k in ["pred_answer", "final_answer", "raw_pred_answer", "prediction"]:
        v = row.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def is_bad_pred(ans: str) -> bool:
    a = normalize_answer(ans)
    return a in {"", "insufficient evidence", "unknown", "none", "not enough information"}


def build_context_from_retrieved(retrieved: List[Dict[str, Any]], topk_docs: int = 5, max_chars: int = 6000) -> str:
    pieces = []
    for d in retrieved[:topk_docs]:
        if not isinstance(d, dict):
            continue
        title = d.get("title", "") or ""
        text = d.get("text", "") or d.get("contents", "") or d.get("content", "") or ""
        title = str(title).strip()
        text = str(text).strip()
        if not title and not text:
            continue
        chunk = f"Title: {title}\n{text}".strip()
        pieces.append(chunk)
    ctx = "\n\n".join(pieces)
    return ctx[:max_chars]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())).strip()


def substring_fallback(pred_answer: str, context: str) -> Tuple[str, float, int, float]:
    if context and pred_answer:
        pa = normalize_text(pred_answer)
        if pa not in {"", "insufficient evidence", "unknown", "none", "not enough information"}:
            if pa in normalize_text(context):
                return pred_answer, 1.0, 1, 1.0
    return "", 0.0, 0, 0.0


def answer_question(
    model,
    tokenizer,
    question: str,
    context: str,
    device: torch.device,
    max_length: int = 384,
    doc_stride: int = 128,
    max_answer_len: int = 30,
) -> Tuple[str, float]:
    """
    Sliding-window extractive QA.
    Returns (best_answer, best_score).

    Matches your working add_qa_verifier_spanmatch.py logic.
    """
    enc = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        padding=False,
        return_tensors=None,
    )

    best_text = ""
    best_score = float("-inf")
    num_chunks = len(enc["input_ids"])

    for i in range(num_chunks):
        input_ids = torch.tensor([enc["input_ids"][i]], device=device)
        attention_mask = torch.tensor([enc["attention_mask"][i]], device=device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits = out.start_logits[0].detach().cpu().numpy()
        end_logits = out.end_logits[0].detach().cpu().numpy()

        if np.isnan(start_logits).any() or np.isnan(end_logits).any():
            continue

        seq_ids = enc.sequence_ids(i)
        context_positions = [j for j, sid in enumerate(seq_ids) if sid == 1]
        if not context_positions:
            continue

        for s in context_positions:
            max_e = min(s + max_answer_len - 1, context_positions[-1])
            for e in range(s, max_e + 1):
                if seq_ids[e] != 1:
                    continue

                span_ids = enc["input_ids"][i][s:e+1]
                text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
                if not text:
                    continue

                text_norm = text.lower().strip()
                if text_norm in {"", "[cls]", "[sep]"}:
                    continue
                if len(text_norm) <= 1:
                    continue

                score = float(start_logits[s] + end_logits[e])
                if score > best_score:
                    best_score = score
                    best_text = text

    if best_score == float("-inf"):
        return "", 0.0
    return best_text, best_score


class QAVerifier:
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad", device: int = 0):
        self.model_name = model_name
        if torch.cuda.is_available() and device >= 0:
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def verify(
        self,
        question: str,
        pred_answer: str,
        retrieved: List[Dict[str, Any]],
        topk_docs: int = 5,
        max_context_chars: int = 6000,
    ) -> Tuple[str, float, int, float]:
        pred_answer = str(pred_answer or "").strip()

        if not question or is_bad_pred(pred_answer):
            return "", 0.0, 0, 0.0

        context = build_context_from_retrieved(
            retrieved=retrieved,
            topk_docs=topk_docs,
            max_chars=max_context_chars,
        ).strip()

        if not context:
            return "", 0.0, 0, 0.0

        # cheap exact support fallback first
        fb_answer, fb_score, fb_em, fb_f1 = substring_fallback(pred_answer, context)
        if fb_em == 1:
            return fb_answer, fb_score, fb_em, fb_f1

        qa_answer, qa_score = answer_question(
            model=self.model,
            tokenizer=self.tokenizer,
            question=question,
            context=context,
            device=self.device,
        )

        return (
            qa_answer,
            float(qa_score),
            exact_match(qa_answer, pred_answer),
            token_f1(qa_answer, pred_answer),
        )