import hashlib
import re

_WS = re.compile(r"\s+")

def normalize_whitespace(s: str) -> str:
    return _WS.sub(" ", s).strip()

def stable_doc_key(title: str, text: str) -> str:
    """
    A stable key to deduplicate documents across queries.
    Uses normalized title + normalized text.
    """
    title_n = normalize_whitespace(title).lower()
    text_n  = normalize_whitespace(text).lower()
    h = hashlib.sha1((title_n + "||" + text_n).encode("utf-8")).hexdigest()
    return h