
from __future__ import annotations
from typing import Dict, Any
import numpy as np, pandas as pd
from .utils import aligned_windows

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Created at"] = pd.to_datetime(d["Created at"], errors="coerce")
    g = d.groupby(["Name","customer_id"], dropna=False).agg({
        "Created at":"max",
        "net_sales":"sum",
        "discount_rate":"mean",
        "units_per_order":"sum",
        "Lineitem name":"first",
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
