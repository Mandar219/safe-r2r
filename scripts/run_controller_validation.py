#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional

import yaml
import torch
from tqdm import tqdm
from transformers import pipeline

from safe_r2r.utils.io import load_yaml, ensure_dir
from safe_r2r.llm.base import LLMConfig
from safe_r2r.llm.factory import make_llm
from safe_r2r.generation.prompting import build_rag_prompt
from safe_r2r.generation.postprocess import postprocess_answer
from safe_r2r.evaluation.metrics import exact_match
from safe_r2r.evaluation.token_overlap import precision_recall_f1

from safe_r2r.retrieval.faiss_retriever import FaissRetriever
from safe_r2r.retrieval.ladder import RetrievalLadder, LadderConfig
from safe_r2r.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from safe_r2r.retrieval.compress import ExtractiveCompressor, CompressionConfig

from safe_r2r.controller.scoring import compute_score


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
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


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def normalize_answer(s: str) -> str:
    import re
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match_text(pred: str, gold: str) -> int:
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


def split_sentences(text: str) -> List[str]:
    import re
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _tok(s: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", str(s or "").lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def build_sentence_max_premise(doc_text: str, question: str, answer: str, max_sents: int = 4, max_chars: int = 1600) -> str:
    sents = split_sentences(doc_text)
    if not sents:
        return ""
    qtoks = _tok(question) + _tok(answer)
    scored = []
    for s in sents:
        stoks = _tok(s)
        score = _jaccard(qtoks, stoks)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = " ".join([s for _, s in scored[:max_sents]]).strip()
    return chosen[:max_chars]


def parse_nli_scores(all_scores):
    pe, pc = 0.0, 0.0
    for d in all_scores:
        lab = str(d.get("label", "")).lower()
        sc = float(d.get("score", 0.0))
        if "entail" in lab:
            pe = max(pe, sc)
        elif "contrad" in lab:
            pc = max(pc, sc)
    return pe, pc


def compute_sentence_max_verifier(nli, question: str, pred_answer: str, retrieved: List[Dict[str, Any]], topk_docs: int = 5) -> float:
    if not pred_answer.strip():
        return 0.0
    hyp = f"Question: {question}\nAnswer: {pred_answer}"
    inputs = []
    for d in retrieved[:topk_docs]:
        txt = d.get("text", "") or ""
        if not txt:
            txt = d.get("title", "") or ""
        prem = build_sentence_max_premise(txt, question, pred_answer)
        if prem:
            inputs.append({"text": prem, "text_pair": hyp})
    if not inputs:
        return 0.0

    outs = nli(inputs, padding=True, truncation=True)
    best_margin = -1.0
    for out in outs:
        if isinstance(out, list):
            pe, pc = parse_nli_scores(out)
        else:
            lab = str(out.get("label", "")).lower()
            sc = float(out.get("score", 0.0))
            pe = sc if "entail" in lab else 0.0
            pc = sc if "contrad" in lab else 0.0
        best_margin = max(best_margin, pe - pc)
    m = (best_margin + 1.0) / 2.0
    return max(0.0, min(1.0, m))


def normalize_text(s: str) -> str:
    import re
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_span_hits(pred_answer: str, retrieved: List[Dict[str, Any]], topk: int = 15):
    ans = normalize_text(pred_answer)
    if ans in {"", "insufficient evidence", "unknown", "none", "not enough information"}:
        return 0, 0
    hits = 0
    full = []
    for d in retrieved[:topk]:
        txt = d.get("text", "") or ""
        full.append(txt)
        if ans and ans in normalize_text(txt):
            hits += 1
    concat = normalize_text(" ".join(full))
    span_hit = int(ans in concat) if ans else 0
    return span_hit, hits


def build_context_for_qa(retrieved: List[Dict[str, Any]], topk_docs: int = 5, max_chars: int = 6000) -> str:
    pieces = []
    for d in retrieved[:topk_docs]:
        title = d.get("title", "") or ""
        text = d.get("text", "") or ""
        chunk = f"Title: {title}\n{text}".strip()
        if chunk:
            pieces.append(chunk)
    return "\n\n".join(pieces)[:max_chars]


def qa_substring_fallback(pred_answer: str, context: str):
    ans = normalize_text(pred_answer)
    ctx = normalize_text(context)
    if ans in {"", "insufficient evidence", "unknown", "none", "not enough information"}:
        return "", 0.0, 0, 0.0
    if ans and ans in ctx:
        return pred_answer, 1.0, 1, 1.0
    return "", 0.0, 0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="main pipeline config")
    ap.add_argument("--controller_config", required=True, help="frozen controller config YAML")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary_out", required=True)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--force_rung4_fallback", action="store_true")
    ap.add_argument("--split", type=str, default=None, help="override dataset split")
    ap.add_argument("--device", type=int, default=0, help="GPU id for verifier pipelines; use -1 for CPU")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    controller_cfg = load_yaml(args.controller_config)["controller"]

    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]
    split = args.split if args.split is not None else cfg["dataset"].get("split_valid", "validation")
    queries_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{split}_queries.jsonl'
    if not Path(queries_path).exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    artifacts_dir = cfg["paths"]["artifacts"]
    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval.index"
    ids_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_doc_ids.json"
    meta_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_meta.json"
    doc_lookup_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_trainval_doc_lookup.json'

    retriever = FaissRetriever(index_path, ids_path, meta_path, doc_lookup_path)

    rung_top_k_raw = cfg.get("ladder", {}).get("rung_top_k", {})
    rung_top_k = {int(k): int(v) for k, v in rung_top_k_raw.items()}

    rer_cfg = cfg.get("reranker", {})
    rer_enabled = bool(rer_cfg.get("enabled", False))
    reranker = None
    rerank_faiss_top_n = int(rer_cfg.get("faiss_top_n", 50))
    if rer_enabled:
        reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name=rer_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                batch_size=int(rer_cfg.get("batch_size", 32)),
            )
        )

    comp_cfg = cfg.get("compression", {})
    comp_enabled = bool(comp_cfg.get("enabled", False))
    compressor = None
    if comp_enabled:
        compressor = ExtractiveCompressor(
            CompressionConfig(
                embedder_model_name=comp_cfg.get("embedder_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                batch_size=int(comp_cfg.get("batch_size", 64)),
                max_sents_per_doc=int(comp_cfg.get("max_sents_per_doc", 4)),
                max_words_per_sent=int(comp_cfg.get("max_words_per_sent", 35)),
                max_words_per_doc=int(comp_cfg.get("max_words_per_doc", 140)),
            )
        )

    ladder = RetrievalLadder(
        retriever,
        LadderConfig(rung_top_k=rung_top_k, rerank_faiss_top_n=rerank_faiss_top_n),
        reranker=reranker,
        compressor=compressor,
    )

    llm_cfg = LLMConfig(**cfg["llm"])
    llm = make_llm(llm_cfg)

    # frozen controller config
    rung_order = [int(r) for r in controller_cfg.get("rung_order", [0, 1, 2, 3, 4])]
    taus = {int(k): float(v) for k, v in controller_cfg.get("taus", {}).items()}
    weights = controller_cfg["weights"]
    knobs = controller_cfg.get("feature_knobs", {})
    global_tau = controller_cfg.get("global_controller_tau", None)
    if global_tau is not None:
        global_tau = float(global_tau)

    # verifier pipelines
    nli_model = controller_cfg.get("nli_model", "roberta-large-mnli")
    qa_model = controller_cfg.get("qa_model", "distilbert-base-cased-distilled-squad")

    nli = pipeline(
        "text-classification",
        model=nli_model,
        device=args.device,
        return_all_scores=True,
        batch_size=64,
    )

    # for now keep QA verifier as substring fallback only, since that was the stable signal
    # if you later want full QA model inference in this script, add it here

    out_rows = []

    certified_accepted = []
    answered_rows = []

    for n, ex in enumerate(tqdm(read_jsonl(queries_path), desc="Controller validation")):
        if args.max_examples is not None and n >= int(args.max_examples):
            break

        qid = ex["qid"]
        q = ex["question"]
        gold = ex["answer"]

        selected = None
        rows_by_rung = {}

        for rung in rung_order:
            # retrieval
            t0 = time.perf_counter()
            docs = ladder.retrieve(q, rung=rung)
            t1 = time.perf_counter()

            retrieved_logged = []
            for d in docs:
                txt = getattr(d, "text", "") or ""
                retrieved_logged.append({
                    "doc_id": d.doc_id,
                    "score": float(d.score),
                    "title": d.title,
                    "text": txt,
                    "text_chars": len(txt) if isinstance(txt, str) else 0,
                })

            # prompt
            if rung == 0:
                prompt = (
                    "Answer the question concisely. Use only your internal knowledge.\n\n"
                    "Output Rules (MUST FOLLOW):\n"
                    "1) Output ONLY the final answer as a short span (one line).\n"
                    "2) Do NOT explain.\n"
                    "3) Do NOT add extra words.\n"
                    "4) Do NOT include quotes.\n"
                    "5) Do NOT repeat the question.\n"
                    "6) If yes/no, output exactly: yes OR no\n"
                    "7) If the documents do not contain enough information, output exactly: Insufficient evidence\n\n"
                    f"Question: {q}\n"
                    "Final answer:"
                )
            else:
                prompt = build_rag_prompt(q, docs)

            # generation
            t2 = time.perf_counter()
            resp = llm.generate(prompt)
            t3 = time.perf_counter()

            raw_pred = (resp.get("text") or "").strip()
            pred = postprocess_answer(raw_pred)

            prf = precision_recall_f1(pred, gold)
            em = exact_match(pred, gold)

            retrieval_latency = (t1 - t0) * 1000.0
            meta = resp.get("meta", {}) or {}
            gen_latency = float(meta.get("latency_ms", (t3 - t2) * 1000.0))

            pt = int(meta.get("prompt_tokens", -1))
            ct = int(meta.get("completion_tokens", -1))
            tt = int(meta.get("total_tokens", (pt + ct) if pt >= 0 and ct >= 0 else -1))

            # signals
            verifier_sent_max_margin = compute_sentence_max_verifier(
                nli=nli,
                question=q,
                pred_answer=pred,
                retrieved=retrieved_logged,
                topk_docs=5,
            )

            span_hit, span_doc_hits = compute_span_hits(pred, retrieved_logged, topk=15)

            context = build_context_for_qa(retrieved_logged, topk_docs=5, max_chars=6000)
            qa_answer, qa_score, qa_match_em, qa_match_f1 = qa_substring_fallback(pred, context)

            row = {
                "qid": qid,
                "question": q,
                "gold_answer": gold,
                "raw_pred_answer": raw_pred,
                "pred_answer": pred,
                "rung": int(rung),
                "num_docs": len(retrieved_logged),
                "retrieved": retrieved_logged,
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": gen_latency,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "em": int(em),
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
                "is_insufficient": int(normalize_answer(pred) == "insufficient evidence"),
                "verifier_sent_max_margin": verifier_sent_max_margin,
                "span_hit": span_hit,
                "span_doc_hits": span_doc_hits,
                "qa_verifier_answer": qa_answer,
                "qa_verifier_score": qa_score,
                "qa_match_em": qa_match_em,
                "qa_match_f1": qa_match_f1,
                "llm_backend": llm_cfg.backend,
                "llm_model": llm_cfg.model_name,
            }

            score = compute_score(
                row,
                wq=float(weights["wq"]),
                wv=float(weights["wv"]),
                ws=float(weights["ws"]),
                wi=float(weights["wi"]),
                wl=float(weights["wl"]),
                wt=float(weights["wt"]),
                topk_hit_k=int(knobs.get("top2_hit_k", 2)),
                length_limit=int(knobs.get("length_limit", 5)),
            )
            row["controller_score"] = score
            rows_by_rung[rung] = row

            tau = taus.get(rung, None)
            if tau is not None and score <= tau:
                selected = (rung, tau, row)
                break

        fallback_used = 0
        accepted_certified = 1
        answered = 1

        if selected is None:
            if args.force_rung4_fallback:
                fallback_used = 1
                accepted_certified = 0
                final_rung = max(rung_order)
                final_tau = taus.get(final_rung, None)
                final_row = rows_by_rung[final_rung]
            else:
                answered = 0
                accepted_certified = 0
                out_rows.append({
                    "qid": qid,
                    "question": q,
                    "gold_answer": gold,
                    "answered": 0,
                    "accepted_certified": 0,
                    "selected_rung": None,
                    "fallback_used": 0,
                    "reason": "no_rung_passed",
                })
                continue
        else:
            final_rung, final_tau, final_row = selected

        # optional global controller gate
        if answered == 1 and global_tau is not None and final_row["controller_score"] > global_tau:
            if args.force_rung4_fallback:
                fallback_used = 1
                accepted_certified = 0
                final_rung = max(rung_order)
                final_tau = taus.get(final_rung, None)
                final_row = rows_by_rung[final_rung]
            else:
                answered = 0
                accepted_certified = 0
                out_rows.append({
                    "qid": qid,
                    "question": q,
                    "gold_answer": gold,
                    "answered": 0,
                    "accepted_certified": 0,
                    "selected_rung": None,
                    "fallback_used": 0,
                    "reason": "failed_global_controller_gate",
                })
                continue

        out = {
            "qid": qid,
            "question": q,
            "gold_answer": gold,
            "pred_answer": final_row["pred_answer"],
            "answered": answered,
            "accepted_certified": accepted_certified,
            "selected_rung": final_rung,
            "fallback_used": fallback_used,
            "controller_score": final_row["controller_score"],
            "tau": final_tau,
            "em": final_row["em"],
            "f1": final_row["f1"],
            "qa_match_f1": final_row["qa_match_f1"],
            "qa_match_em": final_row["qa_match_em"],
            "verifier_sent_max_margin": final_row["verifier_sent_max_margin"],
            "span_doc_hits": final_row["span_doc_hits"],
            "prompt_tokens": final_row["prompt_tokens"],
            "completion_tokens": final_row["completion_tokens"],
            "total_tokens": final_row["total_tokens"],
            "retrieval_latency_ms": final_row["retrieval_latency_ms"],
            "generation_latency_ms": final_row["generation_latency_ms"],
        }
        out_rows.append(out)

        if answered == 1:
            answered_rows.append(out)
        if accepted_certified == 1:
            certified_accepted.append(out)

    write_jsonl(args.out, out_rows)

    summary = {
        "num_queries": len(out_rows),
        "num_answered_total": len(answered_rows),
        "answer_coverage": len(answered_rows) / len(out_rows) if out_rows else 0.0,
        "num_certified_accepted": len(certified_accepted),
        "certified_coverage": len(certified_accepted) / len(out_rows) if out_rows else 0.0,
        "num_fallback_answered": sum(int(r.get("fallback_used", 0)) for r in out_rows),
        "num_unanswered": sum(1 for r in out_rows if int(r.get("answered", 0)) == 0),
        "accept_by_rung": {str(r): sum(1 for x in certified_accepted if x.get("selected_rung") == r) for r in rung_order},
        "fallback_by_rung": {str(r): sum(1 for x in out_rows if int(x.get("fallback_used", 0)) == 1 and x.get("selected_rung") == r) for r in rung_order},
        "controller_config": controller_cfg,
        "force_rung4_fallback": bool(args.force_rung4_fallback),
        "split": split,
    }

    if answered_rows:
        ems = np.array([int(r["em"]) for r in answered_rows], dtype=int)
        summary["empirical_accuracy_answered"] = float(np.mean(ems))
        summary["empirical_risk_answered"] = float(np.mean(1 - ems))
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "retrieval_latency_ms", "generation_latency_ms"]:
            vals = [float(r[key]) for r in answered_rows if r.get(key) is not None]
            summary[f"avg_{key}_answered"] = _safe_mean(vals)

    if certified_accepted:
        ems = np.array([int(r["em"]) for r in certified_accepted], dtype=int)
        summary["empirical_accuracy_certified"] = float(np.mean(ems))
        summary["empirical_risk_certified"] = float(np.mean(1 - ems))

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()