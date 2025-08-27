
from __future__ import annotations
import numpy as np

def compute_score(financial, significance, effect_size, confidence, audience_size) -> float:
    return float(0.35*financial + 0.25*significance + 0.20*effect_size + 0.10*confidence + 0.10*audience_size)

def significance_to_score(p, q, alpha):
    if q is None or np.isnan(q): return 0.0
    return float(np.clip(1.0 - q/alpha, 0.0, 1.0))

def effect_to_score(effect_abs, floor):
    if effect_abs is None or np.isnan(effect_abs): return 0.0
    if floor<=0: return float(np.clip(abs(effect_abs), 0.0, 1.0))
    return float(np.clip(abs(effect_abs)/(2*floor), 0.0, 1.0))

def audience_to_score(n, min_n):
    if n<=0: return 0.0
    return float(np.clip((n - min_n)/(4*min_n if min_n>0 else 1), 0.0, 1.0))

def confidence_from_ci(ci_low, ci_high) -> float:
    """
    Map a CI to a confidence score in [0,1].
    - If CI is missing/NaN or straddles 0 -> return a conservative 0.5.
    - If CI excludes 0 -> return a higher confidence (0.9).
    (You can later refine this to use CI width.)
    """
    # Missing CI → neutral confidence
    if ci_low is None or ci_high is None:
        return 0.5
    try:
        lo = float(ci_low)
        hi = float(ci_high)
    except (TypeError, ValueError):
        return 0.5
    if np.isnan(lo) or np.isnan(hi):
        return 0.5
    # Straddles zero → low confidence
    if lo <= 0.0 <= hi:
        return 0.5
    # Excludes zero → higher confidence
    return 0.9

def financial_to_score(expected_lift, floor):
    import numpy as np
    if expected_lift is None or np.isnan(expected_lift) or expected_lift<=0: return 0.0
    return float(np.clip(expected_lift / (3*floor), 0.0, 1.0))
