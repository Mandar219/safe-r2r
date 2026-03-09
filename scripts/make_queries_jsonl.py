from datasets import load_dataset
from tqdm import tqdm
import argparse
from safe_r2r.utils.io import load_yaml, ensure_dir, write_jsonl

def normalize_supporting_facts(sf):
    # HF often stores as dict: {"title": [...], "sent_id": [...]}
    if isinstance(sf, dict) and "title" in sf:
        titles = sf["title"]
        # sometimes key is "sent_id" or "sent_id" equivalent
        sent_key = "sent_id" if "sent_id" in sf else ("sent_id" if "sent_id" in sf else None)
        sent_ids = sf[sent_key] if sent_key else [None] * len(titles)
        return [{"title": t, "sent_id": int(s) if s is not None else None}
                for t, s in zip(titles, sent_ids)]

    # sometimes it’s list of [title, sent_id]
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
            # item may be [title, [sentences...]] or dict
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
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--split", type=str, default="validation", choices=["train","validation"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ds_name = cfg["dataset"]["name"]
    ds_cfg  = cfg["dataset"]["config"]

    split = args.split
    if split is None:
        split = cfg["dataset"]["split_valid"]

    max_n = args.max_examples
    if max_n is None:
        max_n = cfg["dataset"]["max_valid_examples"]

    ensure_dir(cfg["paths"]["data_processed"])

    ds = load_dataset(ds_name, ds_cfg)[split]
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))

    rows = []
    for ex in tqdm(ds, desc=f"Normalizing {ds_name}/{ds_cfg}:{split}"):
        rows.append(normalize_example(ex))

    out_path = f'{cfg["paths"]["data_processed"]}/{ds_name}_{ds_cfg}_{split}_queries.jsonl'
    write_jsonl(out_path, rows)
    print("Wrote:", out_path)
    print("Num queries:", len(rows))

if __name__ == "__main__":
    main()