from typing import Any, Dict, Iterable, Optional, Tuple


DEFAULT_RUNG_ORDER = [0, 1, 2, 3, 4]


def select_first_safe_rung(
    rows_by_rung: Dict[int, Dict[str, Any]],
    taus: Dict[int, float],
    score_fn,
    rung_order: Iterable[int] = DEFAULT_RUNG_ORDER,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Returns:
      (selected_rung, selected_score, selected_tau)

    Picks the first rung whose score <= tau.
    """
    for rung in rung_order:
        row = rows_by_rung.get(rung)
        if row is None:
            continue
        tau = taus.get(rung)
        if tau is None:
            continue
        score = score_fn(row)
        if score <= tau:
            return rung, score, tau
    return None, None, None