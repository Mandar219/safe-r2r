#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from transformers import pipeline


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


# -----------------------------
# Text utils
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return _WORD_RE.findall(s.lower())

def normalize_ws(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter that works ok for Wikipedia-style text.
    """
    text = normalize_ws(text)
    if not text:
        return []
    # Split on ., !, ? followed by space/cap or end
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    # Fallback if nothing splits
    if len(parts) == 1:
        return [text]
    # Filter tiny fragments
    sents = [p.strip() for p in parts if len(p.strip()) >= 10]
    return sents if sents else [text]

def jaccard_overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    A = set(a_tokens)
    B = set(b_tokens)
    if not A or not B:
        return 0.0
    return float(len(A & B)) / float(len(A | B))

def pick_answer(row: Dict[str, Any]) -> str:
    for k in ["pred_answer", "final_answer", "raw_pred_answer", "prediction"]:
        v = row.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def make_hypothesis(question: str, answer: str) -> str:
    q = normalize_ws(question or "")
    a = normalize_ws(answer or "")
    if not a:
        return ""
    # Question-aware hypothesis; works better than "The answer is X"
    return f"Question: {q}\nAnswer: {a}"

def build_premise_sentence_max(
    doc_text: str,
    query_tokens: List[str],
    max_sents: int = 4,
    max_chars: int = 1600,
) -> str:
    """
    Select up to max_sents sentences with highest token overlap with query_tokens,
    then concatenate and truncate to max_chars.
    """
    sents = split_sentences(doc_text)
    if not sents:
        return ""

    scored: List[Tuple[float, str]] = []
    for s in sents:
        stoks = tokenize(s)
        score = jaccard_overlap(query_tokens, stoks)
        # small length bonus to avoid super-short
        score += 0.02 * min(1.0, len(stoks) / 25.0)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [s for _, s in scored[:max_sents]]
    prem = normalize_ws(" ".join(chosen))
    if len(prem) > max_chars:
        prem = prem[:max_chars] + " ..."
    return prem


# -----------------------------
# MNLI parsing
# -----------------------------
def parse_all_scores(all_scores: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    all_scores: list of {'label':..., 'score':...} for entail/neutral/contradiction
    Returns (p_entail, p_contra)
    """
    pe, pc = 0.0, 0.0
    for d in all_scores:
        lab = str(d.get("label", "")).lower()
        sc = float(d.get("score", 0.0))
        if "entail" in lab:
            pe = max(pe, sc)
        elif "contrad" in lab:
            pc = max(pc, sc)
    return pe, pc


def compute_sentence_max_verifier(
    nli,
    row: Dict[str, Any],
    topk_docs: int = 5,
    max_sents_per_doc: int = 4,
    premise_max_chars: int = 1600,
) -> Dict[str, float]:
    """
    Returns dict with:
      - margin_raw in [-1,1]
      - margin_01 in [0,1]
      - entail_max, contra_max
    """
    q = row.get("question", "") or ""
    ans = pick_answer(row)

    hyp = make_hypothesis(q, ans)
    if not hyp:
        return {"margin_raw": -1.0, "margin_01": 0.0, "entail_max": 0.0, "contra_max": 0.0}

    docs = row.get("retrieved", [])
    if not isinstance(docs, list) or not docs:
        return {"margin_raw": -1.0, "margin_01": 0.0, "entail_max": 0.0, "contra_max": 0.0}

    # Build query tokens from BOTH question + answer (this is important)
    qtoks = tokenize(q) + tokenize(ans)
    if not qtoks:
        qtoks = tokenize(q)

    inputs = []
    for d in docs[:topk_docs]:
        if not isinstance(d, dict):
            continue
        txt = d.get("text", "")
        if not isinstance(txt, str) or not txt.strip():
            txt = d.get("title", "") or ""
        prem = build_premise_sentence_max(
            txt,
            qtoks,
            max_sents=max_sents_per_doc,
            max_chars=premise_max_chars,
        )
        prem = prem.strip()
        if prem:
            inputs.append({"text": prem, "text_pair": hyp})

    if not inputs:
        return {"margin_raw": -1.0, "margin_01": 0.0, "entail_max": 0.0, "contra_max": 0.0}

    outs = nli(inputs, padding=True, truncation=True)

    best_margin = -1.0
    best_entail = 0.0
    best_contra = 0.0

    for out in outs:
        if isinstance(out, list):
            pe, pc = parse_all_scores(out)
        else:
            lab = str(out.get("label", "")).lower()
            sc = float(out.get("score", 0.0))
            pe = sc if "entail" in lab else 0.0
            pc = sc if "contrad" in lab else 0.0

        margin = pe - pc
        if margin > best_margin:
            best_margin = margin
            best_entail = pe
            best_contra = pc

    # map [-1,1] -> [0,1]
    margin_01 = (best_margin + 1.0) / 2.0
    margin_01 = max(0.0, min(1.0, margin_01))

    return {"margin_raw": float(best_margin), "margin_01": float(margin_01),
            "entail_max": float(best_entail), "contra_max": float(best_contra)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", type=str, default="roberta-large-mnli")
    ap.add_argument("--topk_docs", type=int, default=5)
    ap.add_argument("--max_sents_per_doc", type=int, default=4)
    ap.add_argument("--premise_max_chars", type=int, default=1600)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)  # GPU -> 0
    args = ap.parse_args()

    rows = list(read_jsonl(args.log))
    print("Rows:", len(rows))

    nli = pipeline(
        "text-classification",
        model=args.model,
        device=args.device,
        return_all_scores=True,
        batch_size=args.batch_size,
    )

    for r in tqdm(rows, desc="NLI sentence-max"):
        out = compute_sentence_max_verifier(
            nli,
            r,
            topk_docs=args.topk_docs,
            max_sents_per_doc=args.max_sents_per_doc,
            premise_max_chars=args.premise_max_chars,
        )
        r["verifier_sent_max_margin"] = out["margin_01"]     # [0,1], higher is better
        r["verifier_sent_max_margin_raw"] = out["margin_raw"]  # [-1,1]
        r["verifier_sent_max_entail"] = out["entail_max"]
        r["verifier_sent_max_contra"] = out["contra_max"]
        r["verifier_sent_topk_docs"] = args.topk_docs
        r["verifier_sent_max_sents_per_doc"] = args.max_sents_per_doc
        r["verifier_sent_model"] = args.model

    write_jsonl(args.out, rows)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()