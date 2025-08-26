
import numpy as np
from src.stats import wilson_ci, two_proportion_test, benjamini_hochberg, mde_proportion, required_n_for_proportion

def test_wilson_ci_basic():
    low, high = wilson_ci(50, 200)
    assert 0.15 < low < 0.35
    assert 0.25 < high < 0.35

def test_two_prop_z_vs_fisher_threshold():
    r = two_proportion_test(60, 300, 45, 300)
    assert abs(r.diff) > 0.0
    assert 0 <= r.p_value <= 1

def test_bh_monotonicity():
    p = [0.001, 0.01, 0.02, 0.5, 0.8]
    q, sig = benjamini_hochberg(p, alpha=0.1)
    assert all(q[i] <= q[i+1] or abs(q[i]-q[i+1])<1e-12 for i in range(len(q)-1))
    assert sig[0] is True

def test_mde_and_required_n():
    mde = mde_proportion(0.2, 1000)
    assert mde > 0
    n_needed = required_n_for_proportion(0.2, 0.02, alpha=0.05, power=0.8)
    assert n_needed > 0
