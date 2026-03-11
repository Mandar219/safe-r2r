#!/usr/bin/env python3
import argparse
from pathlib import Path
import json

from safe_r2r.utils.io import load_yaml, ensure_dir
from safe_r2r.llm.base import LLMConfig
from safe_r2r.llm.factory import make_llm

from safe_r2r.retrieval.faiss_retriever import FaissRetriever
from safe_r2r.retrieval.ladder import RetrievalLadder, LadderConfig
from safe_r2r.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from safe_r2r.retrieval.compress import ExtractiveCompressor, CompressionConfig

from safe_r2r.experiments.run_rung_eval import run_rung


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--out_tag", type=str, default="calib")
    parser.add_argument("--rungs", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    ensure_dir(cfg["paths"]["logs"])
    ensure_dir(cfg["paths"]["artifacts"])

    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]

    # Train calibration subset
    calib_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_train_calib.jsonl'
    if not Path(calib_path).exists():
        raise FileNotFoundError(f"Missing train calib file: {calib_path}")

    # Retriever artifacts
    artifacts_dir = cfg["paths"]["artifacts"]
    index_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval.index"
    ids_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_doc_ids.json"
    meta_path = f"{artifacts_dir}/faiss_{ds_name}_{ds_cfg}_trainval_meta.json"
    doc_lookup_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_trainval_doc_lookup.json'

    retriever = FaissRetriever(index_path, ids_path, meta_path, doc_lookup_path)

    # Reranker
    rer_cfg = cfg.get("reranker", {})
    reranker = None
    rerank_faiss_top_n = int(rer_cfg.get("faiss_top_n", 50))
    if bool(rer_cfg.get("enabled", False)):
        reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name=rer_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                batch_size=int(rer_cfg.get("batch_size", 32)),
            )
        )

    # Compressor
    comp_cfg = cfg.get("compression", {})
    compressor = None
    if bool(comp_cfg.get("enabled", False)):
        compressor = ExtractiveCompressor(
            CompressionConfig(
                embedder_model_name=comp_cfg.get("embedder_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                batch_size=int(comp_cfg.get("batch_size", 64)),
                max_sents_per_doc=int(comp_cfg.get("max_sents_per_doc", 4)),
                max_words_per_sent=int(comp_cfg.get("max_words_per_sent", 35)),
                max_words_per_doc=int(comp_cfg.get("max_words_per_doc", 140)),
            )
        )

    # Ladder config
    rung_top_k_raw = cfg.get("ladder", {}).get("rung_top_k", {})
    rung_top_k = {int(k): int(v) for k, v in rung_top_k_raw.items()}
    ladder = RetrievalLadder(
        retriever,
        LadderConfig(rung_top_k=rung_top_k, rerank_faiss_top_n=rerank_faiss_top_n),
        reranker=reranker,
        compressor=compressor,
    )

    # LLM
    llm_cfg = LLMConfig(**cfg["llm"])
    llm = make_llm(llm_cfg)

    rungs = [int(x.strip()) for x in args.rungs.split(",") if x.strip()]

    for rung in rungs:
        log_path = f'{cfg["paths"]["logs"]}/{args.out_tag}_rung{rung}.jsonl'
        metrics_path = f'{cfg["paths"]["artifacts"]}/{args.out_tag}_rung{rung}_metrics.json'

        print(f"\n=== Calib run: rung {rung} ===")
        metrics = run_rung(
            rung=rung,
            queries_path=calib_path,
            ladder=ladder,
            llm=llm,
            log_path=log_path,
            metrics_path=metrics_path,
            llm_backend=llm_cfg.backend,
            llm_model=llm_cfg.model_name,
            max_examples=args.max_examples,
            config_path=args.config,
        )
        print("Saved metrics:", metrics_path)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()