import re

_PREFIX_RE = re.compile(r'^\s*(final answer|answer)\s*:\s*', flags=re.IGNORECASE)
_NONANSWER_START_RE = re.compile(r'^\s*(here is|the answer is|i think|based on|from the documents)\b', flags=re.IGNORECASE)

def postprocess_answer(text: str) -> str:
    """
    Convert raw LLM output to a short, canonical answer.

    Goals:
    - keep only first meaningful line
    - normalize yes/no
    - normalize "Insufficient evidence"
    - strip common prefixes
    - remove trailing punctuation and quotes
    - hard-cap length to avoid rambly outputs
    """
    if not text:
        return ""

    t = text.strip()

    # Split into non-empty lines; keep first non-empty
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[0] if lines else ""

    # Remove "Answer:" / "Final answer:" prefix
    t = _PREFIX_RE.sub("", t).strip()

    # If model starts with a generic phrase, try next line (if exists)
    if _NONANSWER_START_RE.search(t) and len(lines) > 1:
        t2 = _PREFIX_RE.sub("", lines[1]).strip()
        if t2:
            t = t2

    # Normalize "Insufficient evidence"
    if "insufficient evidence" in t.lower():
        return "Insufficient evidence"

    # Strip wrapping quotes/backticks
    t = t.strip("`\"'“”‘’ ")

    # Yes/No normalization (common failure: "Yes, ..." or "No.")
    m = re.match(r'^(yes|no)\b', t, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # If the model returns "yes." or "no." after stripping, catch it again
    t2 = t.rstrip(".!?;:, ")
    m2 = re.match(r'^(yes|no)\b$', t2, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).lower()

    # Hard-cap answer length (Hotpot answers are short)
    toks = t.split()
    if len(toks) > 20:
        t = " ".join(toks[:20])

    # Remove trailing punctuation after cap
    t = t.rstrip(".!?;:, ")

    return t