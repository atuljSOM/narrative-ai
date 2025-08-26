
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from math import sqrt
from scipy import stats

@dataclass
class ProportionTestResult:
    p1: float; p2: float; n1: int; n2: int
    diff: float; se: float; z: float; p_value: float
    method: str; ci_low: float; ci_high: float

def wilson_ci(successes: int, n: int, conf: float = 0.95) -> Tuple[float,float]:
    if n==0: return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1-conf)/2); ph = successes/n
    denom = 1 + z**2/n
    center = (ph + z**2/(2*n)) / denom
    adj = z * sqrt((ph*(1-ph) + z**2/(4*n))/n) / denom
    return max(0.0, center-adj), min(1.0, center+adj)

def two_proportion_test(x1: int, n1: int, x2: int, n2: int, conf: float=0.95) -> ProportionTestResult:
    p1 = x1/n1 if n1 else np.nan; p2 = x2/n2 if n2 else np.nan
    if min(x1, n1-x1, x2, n2-x2) < 5:
        table = np.array([[x1, n1-x1],[x2, n2-x2]])
        _, p = stats.fisher_exact(table, alternative='two-sided')
        p_pool = (x1+x2)/(n1+n2) if (n1+n2)>0 else np.nan
        se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)) if (n1>0 and n2>0) else np.nan
        zcrit = stats.norm.ppf(1 - (1-conf)/2); diff = (p1 - p2) if (n1>0 and n2>0) else np.nan
        return ProportionTestResult(p1,p2,n1,n2,diff,se,np.nan,p,"fisher", diff - zcrit*se, diff + zcrit*se)
    p_pool = (x1+x2)/(n1+n2); se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)); z = ((p1)-(p2))/se if se>0 else 0.0
    p = 2*(1 - stats.norm.cdf(abs(z))); zcrit = stats.norm.ppf(1 - (1-conf)/2); diff = (p1-p2)
    return ProportionTestResult(p1,p2,n1,n2,diff,se,z,p,"ztest", diff - zcrit*se, diff + zcrit*se)

@dataclass
class MeanTestResult:
    m1: float; m2: float; n1: int; n2: int; diff: float; p_value: float
    method: str; ci_low: float; ci_high: float; cohen_d: float

def bootstrap_mean_ci(a: np.ndarray, b: np.ndarray, n_boot=800, conf=0.95, seed=123):
    rng = np.random.default_rng(seed); diffs=[]
    n1, n2 = len(a), len(b)
    if n1==0 or n2==0: return (np.nan, np.nan)
    for _ in range(n_boot):
        s1 = rng.choice(a, size=n1, replace=True); s2 = rng.choice(b, size=n2, replace=True)
        diffs.append(s1.mean()-s2.mean())
        low = np.percentile(diffs, (1-conf)/2*100); high = np.percentile(diffs, (1+conf)/2*100)
    return float(low), float(high)

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float); b = b.astype(float)
    ma, mb = np.nanmean(a), np.nanmean(b); sa, sb = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    na, nb = len(a), len(b); denom = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2)) if (na+nb-2)>0 else np.nan
    return (ma-mb)/denom if denom>0 else 0.0

def welch_t_test(a: np.ndarray, b: np.ndarray, conf: float=0.95) -> MeanTestResult:
    a = np.asarray(a, float); b = np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
    diff = float(np.nanmean(a) - np.nanmean(b)); ci_low, ci_high = bootstrap_mean_ci(a[~np.isnan(a)], b[~np.isnan(b)], conf=conf)
    return MeanTestResult(float(np.nanmean(a)), float(np.nanmean(b)), len(a), len(b), diff, float(p), "welch", float(ci_low), float(ci_high), float(cohen_d(a,b)))

def benjamini_hochberg(pvals: List[float], alpha: float=0.10):
    p = np.array(pvals, float); n=len(p); order = np.argsort(p); ranked=p[order]
    q = np.empty(n); prev=1.0
    for i in range(n-1, -1, -1):
        q_i = ranked[i]*n/(i+1); prev = min(prev, q_i); q[i]=prev
    out = np.empty(n); out[order]=q
    return out.tolist(), (out<=alpha).tolist()

def mde_proportion(p: float, n: int, alpha=0.05, power=0.8) -> float:
    z_alpha = stats.norm.ppf(1 - alpha/2); z_beta = stats.norm.ppf(power)
    se = np.sqrt(2*p*(1-p)/n) if n>0 else np.nan
    return float((z_alpha+z_beta)*se)

def required_n_for_proportion(p: float, delta: float, alpha=0.05, power=0.8, max_n=100000) -> int:
    if delta<=0 or p<=0 or p>=1: return 0
    n=10
    while n<max_n and mde_proportion(p, n, alpha=alpha, power=power) > delta:
        n = int(n*1.2)+1
    return n
