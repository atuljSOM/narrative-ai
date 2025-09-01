
from __future__ import annotations
from typing import Dict, Any
import numpy as np, pandas as pd
from .utils import aligned_windows, estimate_expected_orders

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Created at"] = pd.to_datetime(d["Created at"], errors="coerce")
    g = d.groupby(["Name","customer_id"], dropna=False).agg({
        "Created at":"max",
        "net_sales":"sum",
        "discount_rate":"mean",
        "units_per_order":"sum",
        "Lineitem name":"first",
        "category":"first",
        "Currency":"first"
    }).reset_index().rename(columns={"Lineitem name":"lineitem_any"})
    g = g.sort_values("Created at")
    g["AOV"] = g["net_sales"]
    g["first_seen"] = g.groupby("customer_id")["Created at"].transform("min")
    g["is_repeat"] = (g["Created at"] > g["first_seen"]).astype(int)
    g["prev_purchase"] = g.groupby("customer_id")["Created at"].shift(1)
    g["days_since_last"] = (g["Created at"] - g["prev_purchase"]).dt.days
    # Robust RFM via percentile ranks
    max_date = pd.to_datetime(g["Created at"]).max()
    g["RecencyDays"] = (max_date - g["Created at"]).dt.days
    rec_rank = g["RecencyDays"].fillna(g["RecencyDays"].max()).rank(pct=True)
    r_quint = pd.cut(rec_rank, bins=[0,.2,.4,.6,.8,1], labels=[5,4,3,2,1], include_lowest=True)
    f_counts = g.groupby("customer_id")["Name"].transform("count"); f_rank = f_counts.rank(pct=True)
    f_quint = pd.cut(f_rank, bins=[0,.2,.4,.6,.8,1], labels=[1,2,3,4,5], include_lowest=True)
    m_rank = g["net_sales"].rank(pct=True); m_quint = pd.cut(m_rank, bins=[0,.2,.4,.6,.8,1], labels=[1,2,3,4,5], include_lowest=True)
    g["R"], g["F"], g["M"] = r_quint.astype(int), f_quint.astype(int), m_quint.astype(int)
    return g

def aligned_periods_summary(g: pd.DataFrame, min_window_n: int = 300) -> Dict[str,Any]:
    max_date = pd.to_datetime(g["Created at"]).max()
    # Try L7, then L28, then L56
    for window in (7, 28, 56):
        s,e,ps,pe = aligned_windows(max_date, window)
        recent = g[(g["Created at"]>=s)&(g["Created at"]<=e)]
        if len(recent) >= min_window_n or window==56:
            prior = g[(g["Created at"]>=ps)&(g["Created at"]<=pe)]
            break
    def rate_repeat(x): return x["is_repeat"].mean() if len(x)>0 else np.nan
    return {
        "window_days": window,
        "recent_start": str(s.date()), "recent_end": str(e.date()),
        "prior_start": str(ps.date()), "prior_end": str(pe.date()),
        "recent_n": int(len(recent)), "prior_n": int(len(prior)),
        "recent_repeat_rate": float(rate_repeat(recent)), "prior_repeat_rate": float(rate_repeat(prior)),
        "recent_aov": float(recent["AOV"].mean()) if len(recent)>0 else np.nan,
        "prior_aov": float(prior["AOV"].mean()) if len(prior)>0 else np.nan,
        "recent_discount_rate": float(recent["discount_rate"].mean()) if len(recent)>0 else np.nan,
        "prior_discount_rate": float(prior["discount_rate"].mean()) if len(prior)>0 else np.nan,
    }


def compute_repeat_curve(g: pd.DataFrame, horizon_days: list[int] = [60, 90], by_category: bool = True):
    """
    Estimate repeat probability and inter-purchase intervals, and derive simple LTV contributions.
    Returns a dict with per-category stats and a per-customer DataFrame of LTV60/LTV90.
    """
    if g is None or g.empty:
        return {"categories": {}, "store": {}, "per_customer": []}
    d = g.copy()
    d["Created at"] = pd.to_datetime(d["Created at"], errors="coerce")
    d = d.dropna(subset=["Created at","customer_id"]) 
    if d.empty:
        return {"categories": {}, "store": {}, "per_customer": []}

    # ensure category exists
    if "category" not in d.columns:
        d["category"] = "unknown"

    # Per-customer sequences
    d = d.sort_values(["customer_id","Created at"]) 
    d["next_ts"] = d.groupby("customer_id")["Created at"].shift(-1)
    d["ipi_days"] = (d["next_ts"] - d["Created at"]).dt.days

    # Compute p_repeat(H): whether a next order occurs within H days
    stats_by_cat: dict = {}
    cats = sorted(d["category"].dropna().unique().tolist())
    for cat in cats:
        dd = d[d["category"] == cat]
        # base AOV per category
        aov_cat = float(dd["net_sales"].mean()) if len(dd) else 0.0
        ipi_med = float(dd["ipi_days"].dropna().median()) if dd["ipi_days"].notna().any() else np.nan
        entry = {}
        for H in horizon_days:
            has_next = (dd["ipi_days"] <= H).fillna(False)
            denom = int(has_next.shape[0])
            p_rep = float(has_next.mean()) if denom > 0 else 0.0
            exp_orders = estimate_expected_orders(int(H), p_rep, float(ipi_med) if not np.isnan(ipi_med) else float(H))
            ltv = float(exp_orders * aov_cat)
            entry[int(H)] = {"p_repeat": p_rep, "median_ipi": None if np.isnan(ipi_med) else float(ipi_med), "aov": aov_cat, "ltv": ltv}
        stats_by_cat[cat] = entry

    # Per-customer LTV using their category at last observed order
    last_order = d.groupby("customer_id").tail(1)
    per_cust = []
    for _, row in last_order.iterrows():
        cat = row.get("category", "unknown")
        ent = stats_by_cat.get(cat) or {}
        rec = {"customer_id": row["customer_id"], "category": cat}
        for H in horizon_days:
            v = ent.get(int(H)) or {}
            rec[f"ltv{int(H)}"] = float(v.get("ltv", 0.0))
        per_cust.append(rec)

    # Store-level aggregate (weighted by orders per category)
    store = {}
    weights = d.groupby("category")["Name"].count().to_dict()
    for H in horizon_days:
        num, den = 0.0, 0.0
        for cat, w in weights.items():
            v = stats_by_cat.get(cat, {}).get(int(H), {})
            num += w * float(v.get("ltv", 0.0))
            den += w
        store[int(H)] = {"ltv": (num/den if den else 0.0)}

    return {"categories": stats_by_cat, "store": store, "per_customer": per_cust}
