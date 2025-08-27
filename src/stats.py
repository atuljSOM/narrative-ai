from __future__ import annotations
from typing import Tuple, List, NamedTuple
import math
import numpy as np
from scipy.stats import norm, fisher_exact, ttest_ind
from math import sqrt, erfc

class TwoProportionResult(NamedTuple):
    diff: float        # p1 - p2 (absolute difference in proportions)
    p_value: float     # two-sided p-value (normal approx)
    ci_low: float      # CI low for (p1 - p2)
    ci_high: float     # CI high for (p1 - p2)

def _zcrit(alpha: float = 0.05) -> float:
    """
    Two-sided z critical value. Precomputed for common alphas.
    Falls back to 1.96 if an uncommon alpha is passed.
    """
    a = float(alpha)
    if abs(a - 0.05) < 1e-9:  # 95% CI
        return 1.959963984540054
    if abs(a - 0.10) < 1e-9:  # 90% CI
        return 1.6448536269514722
    if abs(a - 0.01) < 1e-9:  # 99% CI
        return 2.5758293035489004
    return 1.959963984540054

def _phi(z: float) -> float:
    """Standard normal CDF via erf: Phi(z) = 0.5 * erfc(-z / sqrt(2))."""
    return 0.5 * erfc(-z / sqrt(2.0))

def two_proportion_test(
    x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05, continuity: bool = False
) -> TwoProportionResult:
    """
    Z test for difference in proportions (two-sided), with 1-α CI for (p1 - p2).
    - p1 = x1/n1, p2 = x2/n2
    - p-value uses pooled SE (classical two-proportion z-test).
    - CI uses unpooled SE (Wald CI for difference).
    - 'continuity': optional Yates correction for z numerator (rarely needed here).
    """
    n1 = int(n1); n2 = int(n2)
    x1 = int(x1); x2 = int(x2)
    if n1 <= 0 or n2 <= 0:
        return TwoProportionResult(diff=np.nan, p_value=1.0, ci_low=np.nan, ci_high=np.nan)

    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2

    # pooled proportion for hypothesis test
    p_pool = (x1 + x2) / (n1 + n2)
    se_pooled = sqrt(max(p_pool * (1.0 - p_pool), 0.0) * (1.0 / n1 + 1.0 / n2))

    # continuity correction (optional)
    num = diff
    if continuity:
        # subtract 0.5/n from absolute difference
        cc = 0.5 * (1.0 / n1 + 1.0 / n2)
        num = np.sign(diff) * max(abs(diff) - cc, 0.0)

    # z and two-sided p-value using normal approx
    if se_pooled == 0.0:
        p_value = 1.0
    else:
        z = num / se_pooled
        # two-sided p = 2 * (1 - Phi(|z|)) = erfc(|z|/sqrt(2))
        p_value = float(erfc(abs(z) / sqrt(2.0)))

    # CI for difference uses unpooled SE
    se_unpooled = sqrt(
        max(p1 * (1.0 - p1), 0.0) / n1 + max(p2 * (1.0 - p2), 0.0) / n2
    )
    zc = _zcrit(alpha)
    if se_unpooled == 0.0:
        ci_low = ci_high = diff
    else:
        ci_low = diff - zc * se_unpooled
        ci_high = diff + zc * se_unpooled

    return TwoProportionResult(diff=float(diff), p_value=float(p_value),
                               ci_low=float(ci_low), ci_high=float(ci_high))
# --- end block ---
# ----------------- Proportions: CI and tests -----------------

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    n = max(0, int(n))
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_z_test(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-proportion z-test (two-sided), returns p-value. Falls back to Fisher if near-zero counts."""
    n1, n2 = int(n1), int(n2)
    s1, s2 = int(s1), int(s2)
    if min(n1, n2) == 0:
        return 1.0
    # Fisher fallback if extreme counts
    if min(s1, s2, n1 - s1, n2 - s2) < 5:
        # construct 2x2 table
        a, b = s1, n1 - s1
        c, d = s2, n2 - s2
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return float(p)
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(p)


# ----------------- Means: Welch t-test (basic) -----------------

def welch_t_test(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Welch t-test p-value (uses scipy.stats.ttest_ind with equal_var=False)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return 1.0
    _, p = ttest_ind(x, y, equal_var=False)
    return float(p)


# ----------------- Multiple testing (BH-FDR) -----------------

def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Return BH-adjusted q-values in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    p = np.asarray(p_values, dtype=float)
    q = p * m / ranks
    # enforce monotonicity from the right
    q_sorted = q[order]
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return q.tolist()


# ----------------- Power / MDE helpers -----------------

def needed_n_for_proportion_delta(p_base: float, delta_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Approx per-group sample size to detect an ABSOLUTE delta in proportions with two-sided z-test.
    """
    p = float(max(min(p_base, 1 - 1e-9), 1e-9))
    d = float(abs(delta_abs))
    if d <= 1e-9:
        return 10**9
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    se_needed = d / (z_alpha + z_beta)
    n = 2 * p * (1 - p) / (se_needed**2)
    return int(math.ceil(n))


# Compatibility alias if your code calls this name
def required_n_for_proportion(p_base: float, delta_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    return needed_n_for_proportion_delta(p_base, delta_abs, alpha, power)
