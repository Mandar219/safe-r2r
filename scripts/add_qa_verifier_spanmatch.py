#!/usr/bin/env python3
import argparse
import json
import re
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


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


def build_context(row: Dict[str, Any], topk_docs: int, max_chars: int) -> str:
    docs = row.get("retrieved", [])
    pieces = []
    for d in docs[:topk_docs]:
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

    This version does an exhaustive bounded search over all context token spans,
    instead of relying on top-k start/end heuristics.
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

        # exhaustive bounded search over context spans
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

                # filter obvious junk spans
                if len(text_norm) <= 1:
                    continue

                score = float(start_logits[s] + end_logits[e])
                if score > best_score:
                    best_score = score
                    best_text = text

    if best_score == float("-inf"):
        return "", 0.0

    return best_text, best_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepset/roberta-large-squad2")
    ap.add_argument("--topk_docs", type=int, default=5)
    ap.add_argument("--max_context_chars", type=int, default=6000)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    rows = list(read_jsonl(args.log))
    print("Rows:", len(rows))

    # DEBUG
    # rows = [r for r in rows if r.get("question","").startswith('What is the name of the musical written by this English comedian')]
    # print("Rows after filter:", len(rows))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    model.eval()

    for r in tqdm(rows, desc="QA verifier"):
        question = str(r.get("question", "")).strip()
        pred_answer = pick_pred_answer(r)

        if not question or is_bad_pred(pred_answer):
            r["qa_verifier_answer"] = ""
            r["qa_verifier_score"] = 0.0
            r["qa_match_em"] = 0
            r["qa_match_f1"] = 0.0
            r["qa_verifier_model"] = args.model
            r["qa_verifier_topk_docs"] = args.topk_docs
            continue

        context = build_context(r, topk_docs=args.topk_docs, max_chars=args.max_context_chars)
        context = context.strip()
        if not context.strip():
            r["qa_verifier_answer"] = ""
            r["qa_verifier_score"] = 0.0
            r["qa_match_em"] = 0
            r["qa_match_f1"] = 0.0
            r["qa_verifier_model"] = args.model
            r["qa_verifier_topk_docs"] = args.topk_docs
            continue

        # cheap substring fallback
        def norm(s):
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()

        if context and pred_answer and norm(pred_answer) not in {"", "insufficient evidence", "unknown", "none", "not enough information"}:
            if norm(pred_answer) in norm(context):
                r["qa_verifier_answer"] = pred_answer
                r["qa_verifier_score"] = 1.0
                r["qa_match_em"] = 1
                r["qa_match_f1"] = 1.0
                r["qa_verifier_model"] = "substr-fallback"
                r["qa_verifier_topk_docs"] = args.topk_docs
                continue

        # DEBUG
        if question.startswith('What is the name of the musical written by this English comedian'):
            print("\nDEBUG ROW")
            print("QUESTION:", question)
            print("PRED:", pred_answer)
            print("CONTEXT PREVIEW:", context[:800])
            print("CONTEXT LEN:", len(context))

        qa_answer, qa_score = answer_question(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=context,
            device=device,
        )

        r["qa_verifier_answer"] = qa_answer
        r["qa_verifier_score"] = float(qa_score)
        r["qa_match_em"] = exact_match(qa_answer, pred_answer)
        r["qa_match_f1"] = token_f1(qa_answer, pred_answer)
        r["qa_verifier_model"] = args.model
        r["qa_verifier_topk_docs"] = args.topk_docs

    write_jsonl(args.out, rows)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()