#!/usr/bin/env python3
import argparse
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Iterator, List

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

from safe_r2r.controller.qa_verifier import QAVerifier
from safe_r2r.controller.checkpointing import load_done_qids, flush_many, save_progress_json


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+", text) if p.strip()]


def _tok(s: str) -> List[str]:
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
        txt = d.get("text", "") or d.get("title", "") or ""
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


def write_row(fh, row: Dict[str, Any]):
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--queries_path", required=True, help="Path to controller-calibration jsonl file")
    ap.add_argument("--out_prefix", required=True, help="Prefix for rung log outputs, e.g. logs/online_calib")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--device", type=int, default=0, help="GPU id for verifier models; use -1 for CPU")
    ap.add_argument("--nli_model", type=str, default="roberta-large-mnli")
    ap.add_argument("--qa_model", type=str, default="distilbert-base-cased-distilled-squad")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint_every", type=int, default=500)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

    artifacts_dir = cfg["paths"]["artifacts"]
    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]

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

    nli = pipeline(
        "text-classification",
        model=args.nli_model,
        device=args.device,
        return_all_scores=True,
        batch_size=64,
    )
    qa_verifier = QAVerifier(model_name=args.qa_model, device=args.device)

    rung_paths = {r: f"{args.out_prefix}_rung{r}.jsonl" for r in [0, 1, 2, 3, 4]}
    progress_path = f"{args.out_prefix}_progress.json"

    done_by_rung = {}
    for r, p in rung_paths.items():
        done_by_rung[r] = load_done_qids(p) if args.resume else set()

    out_files = {}
    try:
        for rung in [0, 1, 2, 3, 4]:
            path = rung_paths[rung]
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if args.resume and Path(path).exists() else "w"
            out_files[rung] = open(path, mode, encoding="utf-8")

        processed = 0
        last_qid = None

        for idx, ex in enumerate(tqdm(read_jsonl(args.queries_path), desc="Online controller calibration logs")):
            if args.max_examples is not None and processed >= int(args.max_examples):
                break

            qid = str(ex["qid"])
            q = ex["question"]
            gold = ex["answer"]
            last_qid = qid

            # skip only if all rung outputs already exist for this qid
            if all(qid in done_by_rung[r] for r in [0, 1, 2, 3, 4]):
                continue

            for rung in [0, 1, 2, 3, 4]:
                if qid in done_by_rung[rung]:
                    continue

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

                verifier_sent_max_margin = compute_sentence_max_verifier(
                    nli=nli,
                    question=q,
                    pred_answer=pred,
                    retrieved=retrieved_logged,
                    topk_docs=5,
                )

                span_hit, span_doc_hits = compute_span_hits(pred, retrieved_logged, topk=15)

                qa_answer, qa_score, qa_match_em, qa_match_f1 = qa_verifier.verify(
                    question=q,
                    pred_answer=pred,
                    retrieved=retrieved_logged,
                    topk_docs=5,
                    max_context_chars=6000,
                )

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
                    "nli_model": args.nli_model,
                    "qa_model": qa_verifier.model_name,
                }

                write_row(out_files[rung], row)
                done_by_rung[rung].add(qid)

            processed += 1

            if processed % args.checkpoint_every == 0:
                flush_many(out_files.values())
                completed_all = len(set.intersection(*(done_by_rung[r] for r in [0, 1, 2, 3, 4])))
                save_progress_json(progress_path, {
                    "processed_examples_this_run": processed,
                    "last_qid": last_qid,
                    "completed_all_rungs": completed_all,
                    "resume": bool(args.resume),
                    "queries_path": args.queries_path,
                    "out_prefix": args.out_prefix,
                })

        flush_many(out_files.values())
        completed_all = len(set.intersection(*(done_by_rung[r] for r in [0, 1, 2, 3, 4])))
        save_progress_json(progress_path, {
            "processed_examples_this_run": processed,
            "last_qid": last_qid,
            "completed_all_rungs": completed_all,
            "resume": bool(args.resume),
            "queries_path": args.queries_path,
            "out_prefix": args.out_prefix,
            "done": True,
        })

    finally:
        for fh in out_files.values():
            fh.close()

    print("Wrote online calibration logs with prefix:", args.out_prefix)
    print("Progress file:", progress_path)


if __name__ == "__main__":
    main()