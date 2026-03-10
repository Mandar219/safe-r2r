from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from safe_r2r.utils.io import load_yaml, ensure_dir

def normalize_supporting_facts(sf):
    # HF often stores as dict: {"title": [...], "sent_id": [...]}
    if isinstance(sf, dict) and "title" in sf:
        titles = sf["title"]
        sent_key = "sent_id" if "sent_id" in sf else None
        sent_ids = sf[sent_key] if sent_key else [None] * len(titles)
        return [
            {"title": t, "sent_id": int(s) if s is not None else None}
            for t, s in zip(titles, sent_ids)
        ]

    # sometimes it’s list of [title, sent_id] or list of dicts
    if isinstance(sf, list):
        out = []
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append({"title": item[0], "sent_id": int(item[1])})
            elif isinstance(item, dict) and "title" in item:
                out.append({"title": item["title"], "sent_id": item.get("sent_id")})
        return out

    return None

def normalize_example(ex):
    ctx = ex["context"]

    # HF HotpotQA usually stores context as:
    # ctx["title"] -> list[str]
    # ctx["sentences"] -> list[list[str]]
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        out_ctx = []
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            text = " ".join(sents).strip()
            out_ctx.append({"title": title, "text": text})
    else:
        # fallback if format differs
        out_ctx = []
        for item in ctx:
            if isinstance(item, dict) and "title" in item and "sentences" in item:
                out_ctx.append({"title": item["title"], "text": " ".join(item["sentences"]).strip()})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
                out_ctx.append({"title": title, "text": " ".join(sents).strip()})

    return {
        "qid": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type", None),
        "level": ex.get("level", None),
        "context": out_ctx,
        "supporting_facts": normalize_supporting_facts(ex.get("supporting_facts", None)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, limits examples. If omitted, writes the full split.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ds_name = cfg["dataset"]["name"]
    ds_cfg  = cfg["dataset"]["config"]

    ensure_dir(cfg["paths"]["data_processed"])

    ds = load_dataset(ds_name, ds_cfg)[args.split]
    if args.max_examples is not None:
        ds = ds.select(range(min(int(args.max_examples), len(ds))))

    out_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{args.split}_queries.jsonl'
    n = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc=f"Normalizing {ds_name}/{ds_cfg}:{args.split}"):
            row = normalize_example(ex)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print("Wrote:", out_path)
    print("Num queries:", n)

if __name__ == "__main__":
    main()