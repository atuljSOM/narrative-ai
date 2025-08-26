
from __future__ import annotations
import pandas as pd, numpy as np
from typing import Dict, Any, Tuple
from .utils import winsorize_series, write_json

MONETARY = ["Subtotal","Total Discount","Shipping","Taxes","Total","Lineitem price"]
REQUIRED = [
    "Name","Created at","Lineitem name","Lineitem quantity","Lineitem price",
    "Lineitem discount","Financial Status","Fulfillment Status","Subtotal",
    "Total Discount","Shipping","Taxes","Total","Currency",
    "Customer Email","Billing Name","Shipping Province","Shipping Country"
]

def robust_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in REQUIRED:
        if c not in df.columns: df[c] = np.nan
    for c in MONETARY:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Lineitem quantity"] = pd.to_numeric(df["Lineitem quantity"], errors="coerce")
    df["Created at"] = pd.to_datetime(df["Created at"], errors="coerce", utc=True).dt.tz_localize(None)
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
    return df, qa

def load_csv(path: str, qa_out_path: str|None=None):
    df = robust_read_csv(path); df, qa = preprocess(df)
    if qa_out_path: write_json(qa_out_path, qa)
    return df, qa
