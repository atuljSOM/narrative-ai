
from __future__ import annotations
import pandas as pd, numpy as np
from typing import Dict, Any, Tuple
from .utils import winsorize_series, write_json, load_category_map, dominant_category_for_order

MONETARY = ["Subtotal","Total Discount","Shipping","Taxes","Total","Lineitem price","Lineitem discount"]
REQUIRED = [
    "Name","Created at","Lineitem name","Lineitem quantity","Lineitem price",
    "Lineitem discount","Financial Status","Fulfillment Status","Subtotal",
    "Total Discount","Shipping","Taxes","Total","Currency",
    "Customer Email","Billing Name","Shipping Province","Shipping Country"
]

def robust_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    def _parse_money_series(s: pd.Series | None) -> pd.Series:
        """Make money columns numeric. Survives $, commas, NBSPs, unicode, (negatives)."""
        if s is None:
            return pd.Series(dtype=float)
        raw = s.astype(str)
        # detect parentheses negatives BEFORE strip
        neg_mask = raw.str.contains(r"^\s*\(.*\)\s*$", na=False)
        # strip everything except digits, dot, minus
        cleaned = raw.str.replace(r"[^\d\.\-]", "", regex=True)
        out = pd.to_numeric(cleaned, errors="coerce")
        # apply negative for parentheses cases
        out.loc[neg_mask] = -out.loc[neg_mask].abs()
        return out

    # Ensure required columns exist (fill with NaN if missing)
    for c in REQUIRED:
        if c not in df.columns:
            df[c] = np.nan

    # Parse monetary columns robustly (create all of them if absent)
    for c in MONETARY:
        if c in df.columns:
            df[c] = _parse_money_series(df[c])
        else:
            df[c] = np.nan

    # Quantities → numeric (guard)
    if "Lineitem quantity" in df.columns:
        df["Lineitem quantity"] = pd.to_numeric(df["Lineitem quantity"], errors="coerce")

    # Dates → datetime (guard)
    if "Created at" in df.columns:
        df["Created at"] = pd.to_datetime(df["Created at"], errors="coerce", utc=True).dt.tz_localize(None)
    if "Cancelled at" in df.columns:
        df["Cancelled at"] = pd.to_datetime(df["Cancelled at"], errors="coerce", utc=True).dt.tz_localize(None)

    return df


def preprocess(df: pd.DataFrame):
    qa = {}
    df = df.copy()
    df["customer_id"] = df.apply(lambda r: (str(r.get("Customer Email","") or r.get("Billing Name","") or f"{r.get('Shipping Country','')}_{r.get('Name','')}")).lower(), axis=1)
    for c in MONETARY:
        df[c] = winsorize_series(df[c])
    df["net_sales"] = df["Subtotal"] - df["Total Discount"]
    df["discount_rate"] = (df["Total Discount"] / df["Subtotal"]).replace([np.inf,-np.inf], np.nan)
    df["units_per_order"] = df["Lineitem quantity"]
    mask_refund = df["Financial Status"].astype(str).str.lower().str.contains("refunded|chargeback")
    mask_test = df["Name"].astype(str).str.contains("test", case=False, na=False)
    before = len(df); df = df[~(mask_refund|mask_test)].copy()
    qa["excluded_refund_or_test"] = before - len(df)
    qa["raw_rows"] = int(before); qa["final_rows"] = int(len(df))
    qa["min_date"] = str(pd.to_datetime(df["Created at"]).min())
    qa["max_date"] = str(pd.to_datetime(df["Created at"]).max())

    # --- Category tagging (dominant per order) ---
    try:
        cat_map = load_category_map()
        if cat_map and 'Name' in df.columns:
            cats: dict[str, str] = {}
            for name, rows in df.groupby('Name'):
                cats[name] = dominant_category_for_order(rows, cat_map)
            df['category'] = df['Name'].map(cats).fillna('unknown')
        else:
            df['category'] = 'unknown'
    except Exception:
        df['category'] = 'unknown'
    return df, qa

def load_csv(path: str, qa_out_path: str|None=None):
    df = robust_read_csv(path); df, qa = preprocess(df)
    if qa_out_path: write_json(qa_out_path, qa)
    return df, qa
