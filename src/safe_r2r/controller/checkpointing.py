import json
import os
from pathlib import Path
from typing import Dict, Iterable, Set


def load_done_qids(jsonl_path: str) -> Set[str]:
    path = Path(jsonl_path)
    if not path.exists():
        return set()

    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                qid = row.get("qid")
                if qid is not None:
                    done.add(str(qid))
            except Exception:
                continue
    return done


def flush_fh(fh):
    fh.flush()
    os.fsync(fh.fileno())


def flush_many(file_handles: Iterable):
    for fh in file_handles:
        flush_fh(fh)


def save_progress_json(path: str, payload: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)