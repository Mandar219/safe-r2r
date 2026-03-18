import json, argparse, re
from pathlib import Path

def read_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pick_answer(row):
    for k in ["pred_answer","final_answer","raw_pred_answer","prediction"]:
        v=row.get(k,"")
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=15)
    args=ap.parse_args()

    rows=list(read_jsonl(args.log))
    for r in rows:
        ans = pick_answer(r)
        ansn = norm(ans)
        if ansn in {"insufficient evidence","unknown","none",""} or len(ansn) < 3:
            r["span_hit"] = 0
            r["span_doc_hits"] = 0
            continue

        docs = r.get("retrieved", [])[:args.topk]
        hits = 0
        full = []
        for d in docs:
            txt = d.get("text","")
            full.append(txt if isinstance(txt,str) else "")
            dn = norm(txt if isinstance(txt,str) else "")
            if ansn and ansn in dn:
                hits += 1

        concat = norm(" ".join(full))
        r["span_hit"] = int(ansn in concat)
        r["span_doc_hits"] = int(hits)

    write_jsonl(args.out, rows)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()