
from __future__ import annotations
from typing import Dict, Any
import numpy as np, pandas as pd
from .utils import aligned_windows, estimate_expected_orders, normalize_product_name

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Created at"] = pd.to_datetime(d["Created at"], errors="coerce")
    # Ensure required columns exist for aggregation
    if "category" not in d.columns:
        d["category"] = "unknown"
    # Aggregate to one row per order/customer.
    # Important: net_sales is an order-level value; do NOT sum across duplicated line-item rows.
    g = d.groupby(["Name","customer_id"], dropna=False).agg({
        "Created at": "max",
        "net_sales": "first",
        "discount_rate": "first",
        "units_per_order": "sum",
        "Lineitem name": "first",
        "category": "first",
        "Currency": "first",
    }).reset_index().rename(columns={"Lineitem name":"lineitem_any"})
    g = g.sort_values("Created at")
    # convenient alias for product-dependent utilities
    g["product"] = g["lineitem_any"].astype(str)
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
    # Use nullable integers to tolerate missing values without crashing
    g["R"], g["F"], g["M"] = r_quint.astype("Int64"), f_quint.astype("Int64"), m_quint.astype("Int64")
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


# Phase 0.5: Lightweight g_items builder for product plays
def build_g_items(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-(customer, product) aggregates from an order-level frame.
    Accepts either:
      - df with 'lineitem_any' (from compute_features), or
      - df/orders with 'Lineitem name', or
      - df with 'products_concat' (pipe-delimited product keys) which will be exploded.

    Output columns:
      - customer_id, product_key, orders_product, last_date, median_ipi_days
      - product_key_raw, product_key_base, size_token (additive)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'customer_id','product_key','orders_product','last_date','median_ipi_days',
            'product_key_raw','product_key_base','size_token'
        ])
    d = df.copy()
    # Normalize required columns
    d['Created at'] = pd.to_datetime(d.get('Created at'), errors='coerce')
    d = d.dropna(subset=['Created at'])
    if 'customer_id' not in d.columns:
        return pd.DataFrame(columns=['customer_id','product_key','orders_product','last_date','median_ipi_days','product_key_raw','product_key_base','size_token'])
    # Determine product source
    prod_series = None
    explode = False
    if 'products_concat' in d.columns:
        prod_series = d['products_concat'].astype(str)
        explode = True
    elif 'lineitem_any' in d.columns:
        prod_series = d['lineitem_any'].astype(str)
    elif 'Lineitem name' in d.columns:
        prod_series = d['Lineitem name'].astype(str)
    else:
        return pd.DataFrame(columns=['customer_id','product_key','orders_product','last_date','median_ipi_days','product_key_raw','product_key_base','size_token'])

    tmp = d[[
        c for c in ['customer_id','Name','Created at'] if c in d.columns
    ]].copy()
    if 'Name' not in tmp.columns:
        # Fallback to index if no order key
        tmp['Name'] = d.index.astype(str)
    tmp['product_key'] = prod_series

    if explode:
        # products_concat: split and explode to per-product rows per order
        tmp['product_key'] = tmp['product_key'].fillna('')
        tmp = tmp.assign(product_key=tmp['product_key'].str.split('|')).explode('product_key')
        tmp['product_key'] = tmp['product_key'].astype(str).str.strip()
        tmp = tmp[tmp['product_key'] != '']

    # Normalized fields (raw/base/size)
    tmp['product_key_raw'] = tmp['product_key'].astype(str)
    base_size = tmp['product_key_raw'].apply(normalize_product_name)
    tmp['product_key_base'] = base_size.apply(lambda t: t[0])
    tmp['size_token'] = base_size.apply(lambda t: t[1])

    # Per (customer, product): distinct orders, last date, median IPI
    tmp = tmp.sort_values(['customer_id','product_key','Created at'])
    grp = tmp.groupby(['customer_id','product_key'], dropna=False)
    orders_product = grp['Name'].nunique().rename('orders_product')
    last_date = grp['Created at'].max().rename('last_date')
    # Compute IPI per group
    def _median_ipi(x: pd.Series) -> float:
        diffs = x.sort_values().diff().dt.days.dropna()
        return float(diffs.median()) if not diffs.empty else float('nan')
    ipi = grp['Created at'].apply(_median_ipi).rename('median_ipi_days')
    out = pd.concat([orders_product, last_date, ipi], axis=1).reset_index()
    # Keep normalized columns (first value in group)
    for col in ['product_key_raw','product_key_base','size_token']:
        if col in tmp.columns:
            out = out.merge(tmp.groupby(['customer_id','product_key'])[col].first().reset_index(), on=['customer_id','product_key'], how='left')
    # Optional base-level counts windows (28d/90d)
    try:
        maxd = pd.to_datetime(d['Created at']).max()
        for H, colname in [(28,'counts_28d_base'), (90,'counts_90d_base')]:
            start = maxd - pd.Timedelta(days=H)
            ww = tmp[(tmp['Created at'] >= start)]
            cnt = (ww.groupby(['customer_id','product_key_base'])['Name'].nunique().rename(colname).reset_index())
            out = out.merge(cnt, left_on=['customer_id','product_key_base'], right_on=['customer_id','product_key_base'], how='left')
    except Exception:
        pass
    return out
