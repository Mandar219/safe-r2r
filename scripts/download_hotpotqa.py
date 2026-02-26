from datasets import load_dataset
from src.utils.io import load_yaml

def main():
    cfg = load_yaml("configs/default.yaml")
    ds_name = cfg["dataset"]["name"]
    ds_cfg = cfg["dataset"]["config"]

    ds = load_dataset(ds_name, ds_cfg)
    print(ds)

    ex = ds["validation"][0]
    print("\nKeys:", list(ex.keys()))
    for k in ["id", "question", "answer", "type", "level"]:
        print(f"{k}: {ex.get(k)}")

    ctx = ex["context"]
    print("\ncontext type:", type(ctx))
    # Case 1: dict-like with 'title' and 'sentences'
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        print("context keys:", list(ctx.keys()))
        print("num paragraphs:", len(ctx["title"]))
        print("title[0]:", ctx["title"][0])
        print("sentences[0][:2]:", ctx["sentences"][0][:2])
    # Case 2: list of [title, sentences] pairs
    elif isinstance(ctx, list) and len(ctx) > 0:
        print("context[0] type:", type(ctx[0]))
        print("context[0]:", ctx[0])
    else:
        print("Unexpected context format:", ctx)

    sf = ex["supporting_facts"]
    print("\nsupporting_facts type:", type(sf))

    if isinstance(sf, dict):
        print("supporting_facts keys:", list(sf.keys()))
        # Common HF format: {"title": [...], "sent_id": [...]}
        if "title" in sf:
            print("supporting_facts title[:2]:", sf["title"][:2])
        if "sent_id" in sf:
            print("supporting_facts sent_id[:2]:", sf["sent_id"][:2])
        # If keys differ, just print first 2 values of each key
        for k, v in sf.items():
            if isinstance(v, list):
                print(f"{k}[:2]:", v[:2])
    else:
        # If it's a list-of-pairs
        try:
            print("supporting_facts[:2]:", sf[:2])
        except Exception as e:
            print("Could not slice supporting_facts:", e)
            print("supporting_facts raw:", sf)

if __name__ == "__main__":
    main()