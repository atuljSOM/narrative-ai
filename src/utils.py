from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd
import re
from .stats import welch_t_test, two_proportion_test, benjamini_hochberg, seasonal_adjustment

# ---------------- Vertical config (Beauty/Supplements/Mixed) ----------------
# These tune audience windows and subscription rules per vertical.
VERTICAL_CONFIG: Dict[str, Dict[str, Any]] = {
    'beauty': {
        'subscription_threshold': 3,           # orders before pushing subscription
        'winback_window': (21, 45),
        'dormant_window': (60, 120),
        'seasonal_adjustment': True,
        'gift_period_detection': True,
        'compliance_tracking': False,
    },
    'supplements': {
        'subscription_threshold': 2,           # push subscription faster
        'winback_window': (35, 50),            # slightly later (after 30-day supply)
        'dormant_window': (45, 90),            # tighter window
        'seasonal_adjustment': False,          # less seasonal
        'gift_period_detection': False,
        'compliance_tracking': True,           # unique to supplements
    },
    'mixed': {                                 # many stores sell both
        'use_product_detection': True,
        'apply_category_rules': True,
        # Fallback windows if product type can't be determined
        'winback_window': (21, 45),
        'dormant_window': (60, 120),
        'subscription_threshold': 3,
        'compliance_tracking': True,
    },
}

def get_vertical_mode() -> str:
    """Return vertical mode from env: 'beauty' | 'supplements' | 'mixed' (default)."""
    v = os.getenv('VERTICAL_MODE') or os.getenv('VERTICAL') or 'mixed'
    v = str(v).strip().lower()
    return v if v in VERTICAL_CONFIG else 'mixed'

def get_vertical(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Get effective vertical configuration, merged into provided cfg if present."""
    mode = (cfg or {}).get('VERTICAL_MODE') or get_vertical_mode()
    return VERTICAL_CONFIG.get(str(mode).lower(), VERTICAL_CONFIG['mixed'])

def categorize_product(product_name: str) -> tuple[str, int]:
    """Returns (category, typical_days_supply) based on basic token rules.

    Categories: 'supplement' | 'skincare' | 'cosmetics' | 'unknown'
    """
    if not product_name:
        return ('unknown', 60)
    product_lower = str(product_name).lower()

    # Supplements patterns
    if any(term in product_lower for term in ['vitamin', 'supplement', 'protein', 'collagen', 'probiotic', 'omega']):
        if '90' in product_lower or '3 month' in product_lower:
            return ('supplement', 90)
        elif '60' in product_lower or '2 month' in product_lower:
            return ('supplement', 60)
        else:  # Default 30-day supply
            return ('supplement', 30)

    # Beauty patterns
    if any(term in product_lower for term in ['serum', 'cream', 'cleanser', 'moisturizer']):
        return ('skincare', 45)
    if any(term in product_lower for term in ['mascara', 'liner', 'brow']):
        return ('cosmetics', 90)
    if any(term in product_lower for term in ['foundation', 'concealer']):
        return ('cosmetics', 180)

    return ('unknown', 60)

def supplement_subscription_urgency(customer_data: pd.DataFrame) -> str:
    """Supplements need different subscription push timing.

    Expects columns: 'Created at' (datetime), 'product' (string)
    """
    if customer_data is None or customer_data.empty:
        return "LOW: Not ready"
    last_order = pd.to_datetime(customer_data['Created at'], errors='coerce').max()
    if pd.isna(last_order):
        return "LOW: Not ready"
    days_since = (pd.Timestamp.now() - last_order).days
    product = str(customer_data.get('product', pd.Series([''])).iloc[0])
    product_type, supply_days = categorize_product(product)

    if product_type == 'supplement':
        if days_since >= supply_days - 5:
            return "URGENT: Likely out of product"
        elif days_since >= supply_days - 10:
            return "HIGH: Running low"
        elif customer_data.shape[0] >= 3:
            return "MEDIUM: Good subscription candidate"
    return "LOW: Not ready"

def supplement_compliance_check(customer_orders: pd.DataFrame) -> float:
    """Estimate compliance for supplements (0.5 poor, 0.75 ok, 1.0 good).

    Expects orders for a single customer/product with 'Created at' and 'product'.
    If not a supplement, returns 1.0.
    """
    if customer_orders is None or customer_orders.empty:
        return 1.0
    dd = customer_orders.copy()
    dd['Created at'] = pd.to_datetime(dd['Created at'], errors='coerce')
    dd = dd.sort_values('Created at')
    if dd.empty:
        return 1.0
    product = str(dd.get('product', pd.Series([''])).iloc[0])
    product_type, supply_days = categorize_product(product)
    if product_type != 'supplement':
        return 1.0
    intervals = dd['Created at'].diff().dt.days
    if intervals.dropna().empty:
        return 1.0
    expected_interval = int(supply_days)
    actual_median = float(pd.to_numeric(intervals, errors='coerce').median())
    if actual_median > expected_interval * 1.5:
        return 0.5
    elif actual_median > expected_interval * 1.2:
        return 0.75
    else:
        return 1.0

def subscription_threshold_for_product(product_name: str, cfg: Dict[str, Any] | None = None) -> int:
    """Return per-product subscription threshold (orders) using vertical + product detection."""
    vmode = (cfg or {}).get('VERTICAL_MODE') or get_vertical_mode()
    v = VERTICAL_CONFIG.get(str(vmode).lower(), VERTICAL_CONFIG['mixed'])
    if str(vmode).lower() == 'mixed' and v.get('use_product_detection', False):
        ptype, _ = categorize_product(product_name or '')
        if ptype == 'supplement':
            return 2
        return 3
    # pure verticals
    return int(v.get('subscription_threshold', 3))

DEFAULTS: Dict[str, Any] = {
    # thresholds & knobs (sane defaults; .env can override)
    "MIN_N_WINBACK": 150,
    "MIN_N_SKU": 60,
    "AOV_EFFECT_FLOOR": 0.03,
    "REPEAT_PTS_FLOOR": 0.02,
    "DISCOUNT_PTS_FLOOR": 0.03,
    "FDR_ALPHA": 0.10,
    "FINANCIAL_FLOOR": 300.0,           # used when FINANCIAL_FLOOR_MODE=fixed
    "FINANCIAL_FLOOR_MODE": "auto",     # auto|fixed
    "FINANCIAL_FLOOR_FIXED": 300.0,
    "GROSS_MARGIN": 0.70,
    "EFFORT_BUDGET": 8,
    # adaptive window policy
    "WINDOW_POLICY": "auto",            # auto|l7|l28|l56
    "L7_MIN_ORDERS": 150,
    "L28_MIN_ORDERS": 250,
    # pilot knobs
    "PILOT_AUDIENCE_FRACTION": 0.2,
    "PILOT_BUDGET_CAP": 200.0,
    # seasonality knobs
    "SEASONAL_ADJUST": True,
    "SEASONAL_PERIOD": 7,
    # display/vertical knobs (read also via os.getenv in components)
    "VERTICAL_MODE": "mixed",     # beauty|supplements|mixed
    "CHARTS_MODE": "detailed",    # detailed|compact
    "SHOW_L7": True,               # show L7 KPI card
}


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce(k: str, v: str) -> Any:
    # Strip inline comments and quotes
    if isinstance(v, str):
        v = v.split('#', 1)[0].strip().strip('"').strip("'")

    if k in {"MIN_N_WINBACK", "MIN_N_SKU", "EFFORT_BUDGET", "L7_MIN_ORDERS", "L28_MIN_ORDERS"}:
        return int(float(v))
    if k in {"AOV_EFFECT_FLOOR", "REPEAT_PTS_FLOOR", "DISCOUNT_PTS_FLOOR", "FDR_ALPHA", "GROSS_MARGIN",
             "FINANCIAL_FLOOR", "FINANCIAL_FLOOR_FIXED", "PILOT_AUDIENCE_FRACTION", "PILOT_BUDGET_CAP"}:
        return float(v)
    if k in {"SEASONAL_ADJUST", "SHOW_L7"}:
        return _parse_bool(v)
    if k in {"WINDOW_POLICY", "FINANCIAL_FLOOR_MODE", "CHARTS_MODE", "VERTICAL_MODE"}:
        return str(v).strip().lower()
    return v


def get_config(env_path: str | None = None) -> Dict[str, Any]:
    """
    Load defaults and override with .env if present.
    .env format: KEY=VALUE per line; ignores comments and blanks.
    """
    cfg = dict(DEFAULTS)
    # Resolve .env path
    env_file = env_path or str(Path(".env"))
    if Path(env_file).exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t or t.startswith("#") or "=" not in t:
                    continue
                k, v = t.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in DEFAULTS:
                    cfg[k] = _coerce(k, v)

    # Allow environment variables to override (useful for containers)
    for k in DEFAULTS.keys():
        if k in os.environ:
            cfg[k] = _coerce(k, os.environ[k])

    # Attach vertical mode + config
    if not cfg.get('VERTICAL_MODE'):
        cfg['VERTICAL_MODE'] = get_vertical_mode()
    cfg['VERTICAL'] = get_vertical(cfg)
    return cfg


def safe_make_dirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_yaml(path: str) -> dict:
    try:
        import yaml
    except Exception:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def load_category_map(default_dir: Optional[str] = None) -> dict[str, list[str]]:
    """
    Load token lists per category from templates/category_map.yml.
    Returns {category: [token,...]}, all lowercased.
    """
    base = Path(default_dir) if default_dir else Path(__file__).resolve().parent.parent / 'templates'
    yml = base / 'category_map.yml'
    data = read_yaml(str(yml)) if yml.exists() else {}
    out: dict[str, list[str]] = {}
    for cat, tokens in (data or {}).items():
        try:
            out[str(cat).lower()] = [str(t).lower() for t in (tokens or [])]
        except Exception:
            continue
    return out


def dominant_category_for_order(rows: pd.DataFrame, cat_map: dict[str, list[str]]) -> str:
    """
    Given all line-items (rows) for a single order, choose a dominant category by token match.
    """
    if not cat_map:
        return 'unknown'
    counts: dict[str, int] = {k: 0 for k in cat_map.keys()}
    token_to_cat: list[tuple[str, str]] = []
    for c, toks in cat_map.items():
        for t in toks:
            token_to_cat.append((t, c))
    # compile simple word tokenizer
    word_re = re.compile(r"[A-Za-z]+")
    names = rows.get('Lineitem name') if 'Lineitem name' in rows.columns else None
    if names is None:
        return 'unknown'
    for name in names.astype(str).str.lower().tolist():
        for w in word_re.findall(name):
            for t, c in token_to_cat:
                if t in w:
                    counts[c] = counts.get(c, 0) + 1
    # pick max count category
    best = 'unknown'
    best_ct = 0
    for c, ct in counts.items():
        if ct > best_ct:
            best, best_ct = c, ct
    return best if best_ct > 0 else 'unknown'


def estimate_expected_orders(H: int, p_repeat: float, median_ipi: float) -> float:
    """Simple expected orders over horizon H using repeat probability and median IPI."""
    H = int(max(H, 0))
    p = float(max(min(p_repeat or 0.0, 1.0), 0.0))
    ipi = float(max(median_ipi or 0.0, 1.0))
    return float(p * (H / ipi))


def choose_window(l7_orders: int, l28_orders: int, policy: str = "auto") -> str:
    """
    Choose analysis window based on volume.
    """
    policy = (policy or "auto").lower()
    if policy == "l7":
        return "L7"
    if policy == "l28":
        return "L28"
    if policy == "l56":
        return "L56"
    # auto policy
    l7_min = DEFAULTS["L7_MIN_ORDERS"]
    l28_min = DEFAULTS["L28_MIN_ORDERS"]
    try:
        l7_min = int(os.getenv("L7_MIN_ORDERS", l7_min))
        l28_min = int(os.getenv("L28_MIN_ORDERS", l28_min))
    except Exception:
        pass
    if l7_orders < l7_min:
        return "L28"
    if l28_orders < l28_min:
        return "L56"
    return "L7"


def financial_floor(l28_net_sales: float, gross_margin: float) -> float:
    """
    Adaptive financial floor ≈ 0.5% of L28 net sales × gross margin, floored at $150.
    """
    mode = str(os.getenv("FINANCIAL_FLOOR_MODE", "auto")).lower()
    if mode == "fixed":
        return float(os.getenv("FINANCIAL_FLOOR_FIXED", DEFAULTS["FINANCIAL_FLOOR"]))
    base = 0.005 * max(l28_net_sales, 0.0) * max(gross_margin, 0.0)
    return max(base, 150.0)

def read_json(path: str) -> dict:
    """
    Safe JSON reader. Returns {} if file doesn't exist or is empty/invalid.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def winsorize_series(
    s: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    skipna: bool = True,
) -> pd.Series:
    """
    Clip a numeric series at given lower/upper quantiles (winsorization).
    Keeps index and name. Non-numeric values are ignored (left as-is if possible).
    """
    # Ensure numeric dtype (coerce errors to NaN)
    s_num = pd.to_numeric(s, errors="coerce")
    q_low = s_num.quantile(lower_quantile, interpolation="linear") if skipna else s_num.quantile(lower_quantile)
    q_hi  = s_num.quantile(upper_quantile, interpolation="linear") if skipna else s_num.quantile(upper_quantile)
    clipped = s_num.clip(lower=q_low, upper=q_hi)
    clipped.name = s.name
    return clipped

def aligned_windows(
    anchor_or_df,
    window_days: int | None = None,
    date_col: str = "Created at",
    multi_windows: tuple[int, ...] = (7, 28, 56),
    anchor_ts: datetime | None = None,
):
    """
    Polymorphic helper:

    1) If first arg is a timestamp-like (pd.Timestamp/datetime/str/np.datetime64), and window_days is not None:
       Returns a 4-tuple: (recent_start, recent_end, prior_start, prior_end)

    2) If first arg is a DataFrame:
       Returns a dict with keys 'anchor', 'L7', 'L28', 'L56' (or whatever in multi_windows),
       where each has 'recent' and 'prior' start/end timestamps.
    """
    # ---- Case 1: timestamp-like input -> single-window tuple ----
    ts_like = (pd.Timestamp, datetime, np.datetime64, str)
    if isinstance(anchor_or_df, ts_like):
        anchor = pd.Timestamp(anchor_or_df)
        w = int(window_days or 7)
        recent_end = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        recent_start = anchor.normalize() - pd.Timedelta(days=w - 1)
        prior_end = recent_start - pd.Timedelta(seconds=1)
        prior_start = recent_start - pd.Timedelta(days=w)
        return recent_start, recent_end, prior_start, prior_end

    # ---- Case 2: DataFrame input -> multi-window dict ----
    df = anchor_or_df
    if df is None or getattr(df, "empty", True):
        raise ValueError("aligned_windows: input DataFrame is empty")

    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"aligned_windows: no valid timestamps in column '{date_col}'")

    anchor = pd.Timestamp(anchor_ts) if anchor_ts is not None else s.max()

    out: dict = {"anchor": anchor}
    for w in multi_windows:
        recent_end = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        recent_start = anchor.normalize() - pd.Timedelta(days=w - 1)
        prior_end = recent_start - pd.Timedelta(seconds=1)
        prior_start = recent_start - pd.Timedelta(days=w)
        out[f"L{w}"] = {
            "window_days": w,
            "recent": {"start": recent_start, "end": recent_end},
            "prior": {"start": prior_start, "end": prior_end},
        }
    return out

def normalize_aligned(al: dict) -> dict:
    """
    Returns a nested view of aligned metrics so templates can always use:
      aligned['L7']['net_sales'], aligned['L28']['orders'], etc.

    Works whether the input dict uses nested keys (aligned['L7']['net_sales'])
    or flat keys (aligned['L7_net_sales']).
    """
    def pick(label: str, key: str, default=None):
        # nested path
        nested = (al.get(label) or {}).get(key)
        if nested is not None:
            return nested
        # flat path fallback
        return al.get(f"{label}_{key}", default)

    out = {"anchor": al.get("anchor")}
    for label in ("L7", "L28"):
        out[label] = {
            "net_sales":     pick(label, "net_sales"),
            "orders":        pick(label, "orders"),
            "aov":           pick(label, "aov"),
            "discount_rate": pick(label, "discount_rate"),
            "repeat_share":  pick(label, "repeat_share"),
        }
    # Include window_days if present
    out["L7"]["window_days"]  = pick("L7", "window_days", 7)
    out["L28"]["window_days"] = pick("L28", "window_days", 28)
    return out

def kpi_snapshot_nested(df: pd.DataFrame, anchor_ts: datetime | None = None) -> dict:
    """
    KPI snapshot for your schema (Shopify Orders/Line-items export).
    Columns present:
    Name, Created at, Cancelled at, Lineitem name, Lineitem quantity,
    Lineitem price, Lineitem discount, Financial Status, Fulfillment Status,
    Subtotal, Total Discount, Shipping, Taxes, Total, Currency,
    Customer Email, Billing Name, Shipping Province, Shipping Country
    """

    if df is None or df.empty:
        return {"anchor": None, "L7": {}, "L28": {}}

    d = df.copy()

    # --- Dates + exclusions ---
    d["Created at"] = pd.to_datetime(d["Created at"], errors="coerce")
    d = d.dropna(subset=["Created at"])
    if d.empty:
        return {"anchor": None, "L7": {}, "L28": {}}

    if "Cancelled at" in d.columns:
        canc = pd.to_datetime(d["Cancelled at"], errors="coerce")
        d = d[canc.isna()]

    if "Financial Status" in d.columns:
        d = d[~d["Financial Status"].astype(str).str.contains("refunded", case=False, na=False)]

    # --- Helpers ---
    def to_float(s: pd.Series | None) -> pd.Series:
        """Strip currency/commas/whitespace/parentheses; '' -> NaN -> float."""
        if s is None:
            return pd.Series(dtype=float)
        ss = s.astype(str).str.strip()
        ss = ss.replace({"": np.nan})
        ss = ss.str.replace(r"[,\s\$£€]", "", regex=True)
        neg = ss.str.match(r"^\(.*\)$")
        ss = ss.str.replace(r"[\(\)]", "", regex=True)
        out = pd.to_numeric(ss, errors="coerce")
        out.loc[neg] *= -1
        return out

    def dedup_orders(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop_duplicates(subset=["Name"]) if "Name" in frame.columns else frame

    def orders_in_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        m = (d["Created at"] >= start) & (d["Created at"] <= end)
        return dedup_orders(d.loc[m].copy())

    def lines_in_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        m = (d["Created at"] >= start) & (d["Created at"] <= end)
        return d.loc[m].copy()  # keep all line-items

    anchor = pd.Timestamp(anchor_ts) if anchor_ts is not None else d["Created at"].max()

    def summarize_window(w: int) -> dict:
        recent_start = anchor.normalize() - pd.Timedelta(days=w - 1)
        recent_end   = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

        rec_orders = orders_in_window(recent_start, recent_end)
        orders = int(rec_orders["Name"].nunique()) if "Name" in rec_orders.columns else int(len(rec_orders))

        net_sales = np.nan
        discount_rate = None

        # ---- A) Subtotal - Total Discount ----
        if "Subtotal" in rec_orders.columns:
            subtotal_sum = to_float(rec_orders["Subtotal"]).sum(skipna=True)
            disc_sum = to_float(rec_orders["Total Discount"]).sum(skipna=True) if "Total Discount" in rec_orders.columns else 0.0
            if not np.isnan(subtotal_sum):
                net_sales = float(subtotal_sum - (disc_sum if not np.isnan(disc_sum) else 0.0))
                discount_rate = float((disc_sum if not np.isnan(disc_sum) else 0.0) / (subtotal_sum + 1e-9))

        # ---- B) Total - Shipping - Taxes (if A didn’t work) ----
        if np.isnan(net_sales) and all(c in rec_orders.columns for c in ["Total", "Shipping", "Taxes"]):
            tot = to_float(rec_orders["Total"]).sum(skipna=True)
            ship = to_float(rec_orders["Shipping"]).sum(skipna=True)
            tax = to_float(rec_orders["Taxes"]).sum(skipna=True)
            if not np.isnan(tot):
                net_sales = float(tot - (ship if not np.isnan(ship) else 0.0) - (tax if not np.isnan(tax) else 0.0))
                # discount_rate unknown on this path → leave None

        # ---- C) Line items: Σ(price*qty) - Σ(lineitem discount) ----
        if np.isnan(net_sales) and all(c in d.columns for c in ["Lineitem price", "Lineitem quantity"]):
            rec_lines = lines_in_window(recent_start, recent_end)
            li_rev = (to_float(rec_lines["Lineitem price"]) * to_float(rec_lines["Lineitem quantity"])).sum(skipna=True)
            li_disc = to_float(rec_lines["Lineitem discount"]).sum(skipna=True) if "Lineitem discount" in rec_lines.columns else 0.0
            if not np.isnan(li_rev):
                net_sales = float(li_rev - (li_disc if not np.isnan(li_disc) else 0.0))
                discount_rate = float(li_disc / (li_rev + 1e-9)) if ("Lineitem discount" in rec_lines.columns and not np.isnan(li_disc)) else None

        aov = float(net_sales / orders) if orders and not np.isnan(net_sales) else None

        # ---- Repeat share (Customer Email) ----
                # ---- Repeat share (robust identity) ----
        repeat_share = None

        # Build a normalized identity key across the FULL dataset (not just the window)
        def customer_key(frame: pd.DataFrame) -> pd.Series:
            # primary: email (lowercase, stripped, empty->NaN)
            if "Customer Email" in frame.columns:
                em = frame["Customer Email"].astype(str).str.strip().str.lower()
                em = em.replace({"": np.nan})
            else:
                em = pd.Series(np.nan, index=frame.index)

            # fallback: name + region for rows with missing email
            name = frame["Billing Name"].astype(str).str.strip().str.lower() if "Billing Name" in frame.columns else ""
            prov = frame["Shipping Province"].astype(str).str.strip().str.lower() if "Shipping Province" in frame.columns else ""
            # if email is NaN, use name|province, else keep email
            fallback = (name + "|" + prov).replace({"|": np.nan})
            key = em.copy()
            key = key.where(key.notna(), fallback)
            return key

        # IMPORTANT: compute first_seen on the unfiltered dataset 'd' (so earlier refunded/cancelled
        # orders don’t disappear and turn repeats into “new”)
        all_keys = customer_key(d)
        first_seen = (
            pd.DataFrame({"key": all_keys, "ts": d["Created at"]})
              .dropna(subset=["key"])
              .groupby("key")["ts"].min()
        )

        # Keys for recent orders (after your usual dedupe by order Name)
        rec_keys = customer_key(rec_orders).dropna()
        denom = int(rec_keys.shape[0])

        # If we barely have any identified customers in the window, don’t emit a misleading “0.0”
        MIN_IDENTIFIED = 10
        if denom >= MIN_IDENTIFIED:
            repeats = rec_keys.map(first_seen).lt(recent_start).sum()
            repeat_share = float(repeats / denom)
        else:
            repeat_share = None  # not enough identified customers to measure reliably


        return {
            "window_days": w,
            "net_sales": None if np.isnan(net_sales) else net_sales,
            "orders": orders,
            "aov": aov,
            "discount_rate": discount_rate,
            "repeat_share": repeat_share,
        }

    return {
        "anchor": anchor,
        "L7": summarize_window(7),
        "L28": summarize_window(28),
    }

# --- add near other imports ---
from typing import Optional, Tuple
from .stats import welch_t_test, two_proportion_test, benjamini_hochberg

def kpi_snapshot_with_deltas(df: pd.DataFrame, anchor_ts: datetime | None = None,
                             min_identified:int=10, discount_positive_is_bad:bool=True,
                             alpha:float=0.05,
                             seasonally_adjust: bool = False,
                             seasonal_period: int = 7) -> dict:
    """
    Returns nested structure with recent/prior KPIs for L7/L28, deltas and significance.
    Values:
      aligned["L7"]["net_sales"|"orders"|"aov"|"discount_rate"|"repeat_share"]
      aligned["L7"]["prior"][same keys]
      aligned["L7"]["delta"][metric]  -> relative change (e.g., +0.062 = +6.2%)
      aligned["L7"]["p"][metric]      -> p-value where applicable (aov, discount_rate, repeat_share)
      aligned["L7"]["sig"][metric]    -> True if p<=alpha and sample floors OK
    Same for "L28".
    """

    if df is None or df.empty:
        return {"anchor": None, "L7": {}, "L28": {}}

    d = df.copy()
    d["Created at"] = pd.to_datetime(d["Created at"], errors="coerce")
    d = d.dropna(subset=["Created at"])
    if d.empty:
        return {"anchor": None, "L7": {}, "L28": {}}

    # Exclude cancelled/refunded for KPI calc (use raw d for first_seen to avoid censoring)
    d_kpi = d.copy()
    if "Cancelled at" in d_kpi.columns:
        canc = pd.to_datetime(d_kpi["Cancelled at"], errors="coerce")
        d_kpi = d_kpi[canc.isna()]
    if "Financial Status" in d_kpi.columns:
        d_kpi = d_kpi[~d_kpi["Financial Status"].astype(str).str.contains("refunded", case=False, na=False)]

    # brutal money cleaner (same as your robust_read_csv)
    def _money(s: pd.Series | None) -> pd.Series:
        if s is None: return pd.Series(dtype=float)
        raw = s.astype(str)
        neg_mask = raw.str.contains(r"^\s*\(.*\)\s*$", na=False)
        cleaned = raw.str.replace(r"[^\d\.\-]", "", regex=True)
        out = pd.to_numeric(cleaned, errors="coerce")
        out.loc[neg_mask] = -out.loc[neg_mask].abs()
        return out

    def _order_level_net(row: pd.Series) -> Optional[float]:
        # Prefer Subtotal - Total Discount; else Total - Shipping - Taxes; else None (line items handled in aggregate)
        if pd.notna(row.get("Subtotal", np.nan)):
            sub = float(row["Subtotal"]); disc = float(row.get("Total Discount", 0.0) or 0.0)
            return sub - disc
        if pd.notna(row.get("Total", np.nan)):
            tot = float(row["Total"]); ship = float(row.get("Shipping", 0.0) or 0.0); tax = float(row.get("Taxes", 0.0) or 0.0)
            return tot - ship - tax
        return None

    # Pre-coerce monetary columns once for performance
    for c in ["Subtotal", "Total Discount", "Shipping", "Taxes", "Total", "Lineitem price", "Lineitem discount"]:
        if c in d_kpi.columns: d_kpi[c] = _money(d_kpi[c])

    # Per-order netsales (for AOV distribution)
    d_kpi["_order_net"] = d_kpi.apply(_order_level_net, axis=1)
    # Dedupe orders by Name for order-level calcs
    d_kpi_ord = d_kpi.drop_duplicates(subset=["Name"]) if "Name" in d_kpi.columns else d_kpi.copy()

    def _identity_key(frame: pd.DataFrame) -> pd.Series:
        # email primary
        em = frame["Customer Email"].astype(str).str.strip().str.lower() if "Customer Email" in frame.columns else pd.Series(np.nan, index=frame.index)
        em = em.replace({"": np.nan})
        # fallback name|province
        name = frame["Billing Name"].astype(str).str.strip().str.lower() if "Billing Name" in frame.columns else ""
        prov = frame["Shipping Province"].astype(str).str.strip().str.lower() if "Shipping Province" in frame.columns else ""
        fallback = (name + "|" + prov).replace({"|": np.nan})
        key = em.copy()
        key = key.where(key.notna(), fallback)
        return key

    # Use the FULL raw df (d) for first_seen so prior history isn’t censored by refunds
    all_keys = _identity_key(d)
    first_seen = (
        pd.DataFrame({"key": all_keys, "ts": d["Created at"]})
          .dropna(subset=["key"])
          .groupby("key")["ts"].min()
    )

    anchor = pd.Timestamp(anchor_ts) if anchor_ts is not None else d_kpi_ord["Created at"].max()

    # Precompute seasonally adjusted daily series if enabled
    # Use raw df 'd' for adjustment so refunds (negative) are retained in net_sales
    if seasonally_adjust:
        orders_adj_ts, _orders_method = seasonal_adjustment(d, 'orders', seasonal=seasonal_period)
        netsales_adj_ts, _nets_method = seasonal_adjustment(d, 'net_sales', seasonal=seasonal_period)
        seasonal_method = _nets_method or _orders_method or ""
    else:
        orders_adj_ts = None
        netsales_adj_ts = None
        seasonal_method = ""
    def _window(anchor: pd.Timestamp, days:int) -> Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp,pd.Timestamp]:
        recent_end = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        recent_start = recent_end.normalize() - pd.Timedelta(days=days-1)
        prior_end = recent_start - pd.Timedelta(seconds=1)
        prior_start = prior_end.normalize() - pd.Timedelta(days=days-1)
        return recent_start, recent_end, prior_start, prior_end

    def _summarize(days:int) -> dict:
        rs, re, ps, pe = _window(anchor, days)

        rec = d_kpi_ord[(d_kpi_ord["Created at"]>=rs) & (d_kpi_ord["Created at"]<=re)].copy()
        pri = d_kpi_ord[(d_kpi_ord["Created at"]>=ps) & (d_kpi_ord["Created at"]<=pe)].copy()

        # orders
        o1 = int(rec["Name"].nunique()) if "Name" in rec.columns else int(len(rec))
        o0 = int(pri["Name"].nunique()) if "Name" in pri.columns else int(len(pri))

        # order-level net sales (prefer order_net; else line items aggregate)
        def _netsales(frame_orders: pd.DataFrame) -> float:
            if "_order_net" in frame_orders.columns and frame_orders["_order_net"].notna().any():
                return float(frame_orders["_order_net"].dropna().sum())
            # fallback to line items if needed
            if all(c in d_kpi.columns for c in ["Lineitem price","Lineitem quantity"]):
                li = d_kpi[(d_kpi["Created at"]>=frame_orders["Created at"].min()) & (d_kpi["Created at"]<=frame_orders["Created at"].max())]
                rev = (_money(li["Lineitem price"]) * pd.to_numeric(li["Lineitem quantity"], errors="coerce")).sum(skipna=True)
                disc = _money(li["Lineitem discount"]).sum(skipna=True) if "Lineitem discount" in li.columns else 0.0
                return float(rev - (disc if not np.isnan(disc) else 0.0))
            return float("nan")

        ns1 = _netsales(rec); ns0 = _netsales(pri)

        # Optional: override orders/net_sales with seasonally adjusted sums over the window
        if seasonally_adjust and orders_adj_ts is not None and netsales_adj_ts is not None:
            try:
                def _sum_range(ts, start, end):
                    return float(ts.loc[start.normalize():end.normalize()].sum())
                o1_adj = int(round(max(0.0, _sum_range(orders_adj_ts, rs, re))))
                o0_adj = int(round(max(0.0, _sum_range(orders_adj_ts, ps, pe))))
                ns1_adj = max(0.0, _sum_range(netsales_adj_ts, rs, re))
                ns0_adj = max(0.0, _sum_range(netsales_adj_ts, ps, pe))
                # Only override if adjusted series cover the windows
                if o1_adj is not None and o0_adj is not None:
                    o1, o0 = o1_adj, o0_adj
                if ns1_adj is not None and ns0_adj is not None:
                    ns1, ns0 = ns1_adj, ns0_adj
            except Exception:
                # fall back silently on any indexing error
                pass

        aov1 = float(ns1/o1) if (o1 and not np.isnan(ns1)) else None
        aov0 = float(ns0/o0) if (o0 and not np.isnan(ns0)) else None

        # discount rate: total discount / subtotal (when available); else None
        def _disc_rate(frame_orders: pd.DataFrame) -> Optional[float]:
            if "Subtotal" in frame_orders.columns:
                sub = _money(frame_orders["Subtotal"]).sum(skipna=True)
                disc = _money(frame_orders["Total Discount"]).sum(skipna=True) if "Total Discount" in frame_orders.columns else 0.0
                if not np.isnan(sub) and sub>0:
                    return float(disc/(sub+1e-9))
            # fallback: line-item based mean discount share if available
            return None

        dr1 = _disc_rate(rec); dr0 = _disc_rate(pri)

        # repeat share (identified customers only)
        def _repeat_share(frame_orders: pd.DataFrame) -> Tuple[Optional[float], int]:
            keys = _identity_key(frame_orders).dropna()
            denom = int(keys.shape[0])
            if denom < min_identified:
                return None, denom
            repeats = keys.map(first_seen).lt(rs).sum()
            return float(repeats/denom), denom

        rr1, id1 = _repeat_share(rec)
        rr0, id0 = _repeat_share(pri)

        # deltas (relative). If prior is None/0/NaN → None
        def _rel_delta(x1, x0):
            if x1 is None or x0 is None: return None
            if isinstance(x1,float) and (np.isnan(x1) or np.isnan(x0)): return None
            if x0 == 0: return None
            return float((x1 - x0) / abs(x0))

        delta = {
            "net_sales": _rel_delta(ns1, ns0),
            "orders":    _rel_delta(o1, o0),
            "aov":       _rel_delta(aov1, aov0),
            "discount_rate": _rel_delta(dr1, dr0),
            "repeat_share":  _rel_delta(rr1, rr0),
        }

        # significance (where it makes sense)
        p = {"aov": None, "discount_rate": None, "repeat_share": None}
        # AOV: Welch t on per-order net values
        a1 = rec["_order_net"].dropna().values if "_order_net" in rec.columns else np.array([])
        a0 = pri["_order_net"].dropna().values if "_order_net" in pri.columns else np.array([])

        def _extract_p(val):
            # accept float, dict-like, or object with attribute
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                v = val.get("p_value")
                return float(v) if v is not None else None
            v = getattr(val, "p_value", None)
            return float(v) if v is not None else None

        if len(a1) > 1 and len(a0) > 1:
            mt = welch_t_test(a1, a0)
            p_aov = _extract_p(mt)
            p["aov"] = p_aov


        # Discount: treat as proportion of orders with any discount > 0
        if "Total Discount" in rec.columns and "Total Discount" in pri.columns:
            x1 = int((_money(rec["Total Discount"])>0).sum()); n1 = int(len(rec))
            x0 = int((_money(pri["Total Discount"])>0).sum()); n0 = int(len(pri))
            if n1>0 and n0>0:
                pr = two_proportion_test(x1,n1,x0,n0); p["discount_rate"] = float(pr.p_value)

        # Repeat share: two-proportion on identified customers
        if rr1 is not None and rr0 is not None and id1>=min_identified and id0>=min_identified:
            # reconstruct counts
            x1 = int(round(rr1*id1)); n1 = id1
            x0 = int(round(rr0*id0)); n0 = id0
            pr2 = two_proportion_test(x1,n1,x0,n0); p["repeat_share"] = float(pr2.p_value)

        # FDR on available p's
        # Build p-list (missing -> 1.0 so it won't flag as significant)
        p_list = []
        p_keys = list(p.keys())
        for k in p_keys:
            v = p[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                p_list.append(1.0)
            else:
                p_list.append(float(v))

        # q-values via safe wrapper
        if any(v < 1.0 for v in p_list):
            qvals = _bh_adjust(p_list, alpha=alpha)
            q = {k: float(qvals[i]) if qvals[i] is not None else None for i, k in enumerate(p_keys)}
        else:
            q = {k: None for k in p_keys}

        # significance: prefer q ≤ alpha if available, else p ≤ alpha
        sig = {}
        for i, k in enumerate(p_keys):
            qk = q.get(k)
            pk = p.get(k)
            if qk is not None and not (isinstance(qk, float) and np.isnan(qk)):
                sig[k] = (qk <= alpha)
            else:
                sig[k] = (pk is not None and pk <= alpha)


        sig = {k: (p[k] is not None and p[k] <= alpha) for k in p.keys()}

        return {
            "net_sales": None if np.isnan(ns1) else float(ns1),
            "orders": o1,
            "aov": aov1,
            "discount_rate": dr1,
            "repeat_share": rr1,
            "prior": {
                "net_sales": None if np.isnan(ns0) else float(ns0),
                "orders": o0,
                "aov": aov0,
                "discount_rate": dr0,
                "repeat_share": rr0,
            },
            "delta": delta,
            "p": p,
            "q": q,
            "sig": sig,
            "meta": {
                "identified_recent": id1,
                "identified_prior": id0
            }
        }

    out = {"anchor": anchor, "L7": _summarize(7), "L28": _summarize(28)}
    out["meta"] = {
        "seasonal_adjusted": bool(seasonally_adjust),
        "seasonal_period": int(seasonal_period) if seasonally_adjust else None,
        "seasonal_method": seasonal_method,
    }
    # convenience: top-level recent values for backward compatibility
    for label in ("L7","L28"):
        for k in ("net_sales","orders","aov","discount_rate","repeat_share"):
            out[label][k] = out[label].get(k)
    # include direction rule hint for UI: discount is "good when down"
    out["direction"] = {"discount_rate": ("down" if discount_positive_is_bad else "up")}
    return out

# --- Safe BH wrapper + fallback (put near top-level helpers in utils.py) ---

def _bh_fallback(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg q-values (independent/positive dependence). No alpha needed."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q_i = (m / rank) * p_sorted[i]
        prev = min(prev, q_i)
        q_sorted[i] = prev
    q = np.empty(m, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q.tolist()

def _bh_adjust(p_list: list[float], alpha: float = 0.05) -> list[float]:
    """
    Try benjamini_hochberg with keyword, then positional, else fall back to local impl.
    Always returns a list of q-values aligned to p_list.
    """
    try:
        # try keyword
        out = benjamini_hochberg(p_list, alpha=alpha)  # type: ignore[arg-type]
        return list(out[0] if isinstance(out, tuple) else out)
    except TypeError:
        try:
            # try positional alpha
            out = benjamini_hochberg(p_list, alpha)  # type: ignore[misc]
            return list(out[0] if isinstance(out, tuple) else out)
        except Exception:
            # pure p->q implementation (no alpha arg)
            return _bh_fallback(p_list)
