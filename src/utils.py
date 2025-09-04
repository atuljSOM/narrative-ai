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

## Removed unused supplement-specific helpers (subscription urgency/compliance)

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
    # confidence selection mode for actions: conservative|aggressive|learning
    "CONFIDENCE_MODE": "conservative",
    # interactions (env-driven). Example formats:
    #  - JSON: {"discount_hygiene->winback_21_45":0.9, "winback_21_45->dormant_multibuyers_60_120":0.92}
    #  - CSV:  discount_hygiene->winback_21_45:0.9, bestseller_amplify->winback_21_45:0.95
    "INTERACTION_FACTORS": "",
    # Inventory knobs
    "INVENTORY_ENFORCEMENT_MODE": "soft",   # soft|hard
    "INVENTORY_MAX_AGE_DAYS": 7,
    "INVENTORY_SAFETY_STOCK": 0,
    "INVENTORY_LEAD_TIME_DAYS": 14,
    "INVENTORY_SAFETY_Z": 1.64,            # ~90% service level
    # JSON/CSV map: {"subscription_nudge":60,"sample_to_full":45,"default":21}
    "INVENTORY_MIN_COVER_DAYS_MAP": "",
    "INVENTORY_ALLOW_BACKORDER": True,
    # Feature flags (Phase 1 shims)
    "FEATURES_DYNAMIC_PRODUCTS": False,
    # Product normalization (base + size parsing)
    "FEATURES_PRODUCT_NORMALIZATION": False,
}


def normalize_product_name(name: str) -> tuple[str, str]:
    """Return (base_product, size_token) from a product title.

    Examples:
      "Vitamin C Serum 30ml" -> ("vitamin c serum", "30ml")
      "Protein Powder (5 lb)" -> ("protein powder", "5lb")
      "Omega-3 90 ct" -> ("omega-3", "90ct")
    """
    if not isinstance(name, str):
        return ("", "")
    s = name.strip().lower()
    # Remove brackets content that often carries size
    import re
    paren = re.findall(r"\(([^)]+)\)", s)
    size = ""
    # Common size patterns
    patterns = [
        r"\b(\d+\s?ml)\b",
        r"\b(\d+(?:\.\d+)?\s?oz)\b",
        r"\b(\d+\s?lb)s?\b",
        r"\b(\d+\s?g)\b",
        r"\b(\d+\s?kg)\b",
        r"\b(\d+\s?ct)\b",
        r"\b(\d+\s?(?:pack|pk))\b",
        r"\b(\d+\s?(?:day|month))\b",
        r"\b((?:1|1\.7|3\.4)\s?oz)\b",  # common beauty sizes
    ]
    # Check parentheses first
    for p in paren:
        ps = p.strip()
        for pat in patterns:
            m = re.search(pat, ps)
            if m:
                size = m.group(1).replace(" ", "")
                break
        if size:
            break
    # If no size found, scan full string
    if not size:
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                size = m.group(1).replace(" ", "")
                break
    # Remove size token from base
    base = s
    if size:
        base = base.replace(size, "")
    # Remove parentheses and extra spaces/punctuation around sizes
    base = re.sub(r"\([^)]*\)", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return (base, size)


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
    if k in {"SEASONAL_ADJUST", "SHOW_L7", "INVENTORY_ALLOW_BACKORDER"}:
        return _parse_bool(v)
    if k in {"WINDOW_POLICY", "FINANCIAL_FLOOR_MODE", "CHARTS_MODE", "VERTICAL_MODE",
             "INVENTORY_ENFORCEMENT_MODE", "CONFIDENCE_MODE"}:
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
    # Parse interaction factors into structured mapping
    cfg['INTERACTION_FACTORS_PARSED'] = parse_interaction_factors(cfg.get('INTERACTION_FACTORS', ''))
    # Parse inventory cover days map
    cfg['INVENTORY_MIN_COVER_DAYS'] = parse_cover_days_map(cfg.get('INVENTORY_MIN_COVER_DAYS_MAP', ''))
    return cfg


def parse_interaction_factors(value: str | dict | None) -> dict[tuple[str, str], float]:
    """Parse campaign interaction dampening factors from env or dict.
    Accepts:
      - JSON string: {"a->b":0.9, "c->d":0.95} or nested {"a":{"b":0.9}}
      - CSV string:  "a->b:0.9, c->d:0.95"
      - Dict already parsed
    Returns dict with keys (prior, current) -> factor (float in (0,1]).
    """
    out: dict[tuple[str, str], float] = {}
    if not value:
        return out
    try:
        if isinstance(value, dict):
            items = []
            # nested dict case
            for k, v in value.items():
                if isinstance(v, dict):
                    for k2, f in v.items():
                        items.append((str(k), str(k2), float(f)))
                else:
                    # flat key like "a->b"
                    prior, curr = str(k).split("->", 1)
                    items.append((prior.strip(), curr.strip(), float(v)))
            for prior, curr, f in items:
                if f <= 0 or f > 1: continue
                out[(prior, curr)] = float(f)
            return out
        # Try JSON
        import json as _json
        try:
            parsed = _json.loads(str(value))
            return parse_interaction_factors(parsed)
        except Exception:
            pass
        # Fallback: CSV style "a->b:0.9, c->d:0.95"
        s = str(value)
        for part in s.split(','):
            t = part.strip()
            if not t:
                continue
            if ':' not in t or '->' not in t:
                continue
            left, f = t.split(':', 1)
            prior, curr = left.split('->', 1)
            try:
                factor = float(f.strip())
            except Exception:
                continue
            if factor <= 0 or factor > 1:
                continue
            out[(prior.strip(), curr.strip())] = factor
    except Exception:
        return {}
    return out

def get_interaction_factors(cfg: dict) -> dict[tuple[str, str], float]:
    """Return parsed interaction mapping, falling back to a conservative default matrix."""
    parsed = cfg.get('INTERACTION_FACTORS_PARSED') or {}
    if parsed:
        return parsed
    # Defaults used if no env provided
    return {
        ("discount_hygiene", "winback_21_45"): 0.90,
        ("discount_hygiene", "bestseller_amplify"): 0.95,
        ("discount_hygiene", "subscription_nudge"): 0.95,
        ("winback_21_45", "dormant_multibuyers_60_120"): 0.92,
        ("bestseller_amplify", "winback_21_45"): 0.95,
    }

def parse_cover_days_map(value: str | dict | None) -> dict[str, int]:
    """Parse per-play minimum cover days mapping from JSON/CSV or dict.
    Returns dict like {"subscription_nudge": 60, "sample_to_full": 45, "default": 21}.
    """
    out: dict[str, int] = {}
    if not value:
        return out
    try:
        if isinstance(value, dict):
            for k, v in value.items():
                try:
                    out[str(k)] = int(float(v))
                except Exception:
                    continue
            return out
        import json as _json
        try:
            parsed = _json.loads(str(value))
            return parse_cover_days_map(parsed)
        except Exception:
            pass
        # CSV form: key:val, key:val
        s = str(value)
        for part in s.split(','):
            t = part.strip()
            if not t or ':' not in t:
                continue
            k, v = t.split(':', 1)
            try:
                out[k.strip()] = int(float(v.strip()))
            except Exception:
                continue
    except Exception:
        return {}
    return out


def safe_make_dirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON with safe defaults for Pandas/NumPy types.
    - Falls back to str() for objects like pd.Timestamp, Path, etc.
    - Keeps indentation for readability.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

# ---------------- Identity helpers (Phase 0 observability) ---------------- #
def standardize_order_key(df: pd.DataFrame) -> pd.Series:
    """Return a robust order key Series mapped to 'Name' semantics.
    Priority: 'Name' -> 'order_id' -> 'Order ID' -> index as string.
    Does not mutate input; safe for logging/coverage only.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=str)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if 'name' in cols:
        return df[cols['name']].astype(str)
    for k in ('order_id', 'order id', 'order number'):
        if k in cols:
            return df[cols[k]].astype(str)
    return pd.Series(df.index.astype(str), index=df.index)

def standardize_customer_key(df: pd.DataFrame) -> pd.Series:
    """Return a robust customer key Series used for customer-level metrics.
    Priority: email (lowercased, stripped; supports aliases) -> explicit customer_id -> fallback name|province.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=str)
    idx = df.index
    # Accept common aliases for email: 'Customer Email' | 'customer_email' | 'email'
    email_series = None
    for cand in ['Customer Email', 'customer_email', 'email']:
        if cand in df.columns:
            email_series = df[cand]
            break
    em = email_series.astype(str).str.strip().str.lower() if email_series is not None else pd.Series(np.nan, index=idx)
    em = em.replace({'': np.nan})
    cid = df['customer_id'].astype(str).str.strip().str.lower() if 'customer_id' in df.columns else pd.Series(np.nan, index=idx)
    cid = cid.replace({'': np.nan})
    name = df['Billing Name'].astype(str).str.strip().str.lower() if 'Billing Name' in df.columns else pd.Series('', index=idx)
    prov = df['Shipping Province'].astype(str).str.strip().str.lower() if 'Shipping Province' in df.columns else pd.Series('', index=idx)
    fallback = (name.fillna('') + '|' + prov.fillna('')).replace({'|': np.nan}).infer_objects(copy=False)
    key = em.where(em.notna(), cid.where(cid.notna(), fallback))
    return key

def identity_coverage(df: pd.DataFrame) -> dict:
    """Basic coverage indicators for identities and product presence.
    Returns: {customer_key_coverage: float, order_key_coverage: float}
    """
    if df is None or df.empty:
        return {"customer_key_coverage": 0.0, "order_key_coverage": 0.0}
    ck = standardize_customer_key(df)
    ok = standardize_order_key(df)
    ck_cov = float((ck.notna() & (ck.astype(str).str.strip() != '')).mean()) if len(ck) else 0.0
    ok_cov = float((ok.notna() & (ok.astype(str).str.strip() != '')).mean()) if len(ok) else 0.0
    return {"customer_key_coverage": ck_cov, "order_key_coverage": ok_cov}


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

## normalize_aligned removed (engine uses src/action_engine.py:_normalize_aligned)

## kpi_snapshot_nested removed (superseded by kpi_snapshot_with_deltas)

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
    
    FIXED: Seasonal adjustments are now stored in metadata, never override actual metrics.
    
    Values:
      aligned["L7"]["net_sales"|"orders"|"aov"|"discount_rate"|
                    "repeat_rate_within_window"|"returning_customer_share"|"new_customer_rate"|
                    "repeat_share" (alias) | "repeat_rate" (alias) | "returning_rate" (alias)]
      aligned["L7"]["prior"][same keys]
      aligned["L7"]["delta"][metric]  -> relative change (e.g., +0.062 = +6.2%)
      aligned["L7"]["p"][metric]      -> p-value where applicable (aov, discount_rate, repeat_share)
      aligned["L7"]["sig"][metric]    -> True if p<=alpha and sample floors OK
      aligned["L7"]["seasonal_expected"] -> Dict with expected values if seasonal adjustment enabled
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
        # Prefer Subtotal - Total Discount; else Total - Shipping - Taxes; else None
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
        """Construct a robust customer identity key.
        Priority: Customer Email, then explicit customer_id if present,
        then Billing Name | Shipping Province fallback. Returns a Series aligned to frame.index.
        """
        idx = frame.index
        # email (primary)
        em = frame["Customer Email"].astype(str).str.strip().str.lower() if "Customer Email" in frame.columns else pd.Series(np.nan, index=idx)
        em = em.replace({"": np.nan})
        # explicit customer_id (secondary)
        cid = frame["customer_id"].astype(str).str.strip().str.lower() if "customer_id" in frame.columns else pd.Series(np.nan, index=idx)
        cid = cid.replace({"": np.nan})
        # name/province fallback
        name_s = frame["Billing Name"].astype(str).str.strip().str.lower() if "Billing Name" in frame.columns else pd.Series("", index=idx)
        prov_s = frame["Shipping Province"].astype(str).str.strip().str.lower() if "Shipping Province" in frame.columns else pd.Series("", index=idx)
        fallback = name_s.fillna("") + "|" + prov_s.fillna("")
        # Avoid FutureWarning for silent downcasting; keep types stable
        fallback = fallback.replace({"|": np.nan}).infer_objects(copy=False)
        key = em.where(em.notna(), cid.where(cid.notna(), fallback))
        return key

    # Use the FULL raw df (d) for first_seen so prior history isn't censored by refunds
    all_keys = _identity_key(d)
    first_seen = (
        pd.DataFrame({"key": all_keys, "ts": d["Created at"]})
          .dropna(subset=["key"])
          .groupby("key")["ts"].min()
    )

    anchor = pd.Timestamp(anchor_ts) if anchor_ts is not None else d_kpi_ord["Created at"].max()

    # Precompute seasonally adjusted daily series if enabled
    # Store these for REFERENCE only - never override actual metrics
    seasonal_data = {}
    if seasonally_adjust:
        orders_adj_ts, _orders_method = seasonal_adjustment(d, 'orders', seasonal=seasonal_period)
        netsales_adj_ts, _nets_method = seasonal_adjustment(d, 'net_sales', seasonal=seasonal_period)
        seasonal_method = _nets_method or _orders_method or ""
        seasonal_data = {
            'orders_ts': orders_adj_ts,
            'netsales_ts': netsales_adj_ts,
            'method': seasonal_method
        }
    else:
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

        # ACTUAL orders - what really happened
        o1_actual = int(rec["Name"].nunique()) if "Name" in rec.columns else int(len(rec))
        o0_actual = int(pri["Name"].nunique()) if "Name" in pri.columns else int(len(pri))

        # Canonical net sales + debug of alternative methods
        def _netsales_debug(frame_orders: pd.DataFrame) -> dict:
            """
            Always use the same canonical method for net sales and record which method was used:
            - Primary: sum of per-order net ("_order_net") which is computed as Subtotal-Discount,
              falling back to Total-Shipping-Taxes at the per-order level.
            - Fallback: line-items aggregation restricted to the same orders/time window.

            Returns a dict with keys: value, method, alt (dict of alternative computations).
            """
            out: dict = {"value": float("nan"), "method": None, "alt": {}}
            # Canonical: per-order net if available
            if "_order_net" in frame_orders.columns and frame_orders["_order_net"].notna().any():
                val = float(frame_orders["_order_net"].dropna().sum())
                out["value"] = val
                out["method"] = "order_net"
            else:
                # Fallback: aggregate line items per order, restricted to the same orders
                val = float("nan")
                if all(c in d_kpi.columns for c in ["Lineitem price", "Lineitem quantity"]):
                    if "Name" in frame_orders.columns and "Name" in d_kpi.columns:
                        order_names = set(frame_orders["Name"].astype(str))
                        li = d_kpi[d_kpi["Name"].astype(str).isin(order_names)].copy()
                    else:
                        # last resort: time-range filter
                        li = d_kpi[(d_kpi["Created at"] >= frame_orders["Created at"].min()) & (d_kpi["Created at"] <= frame_orders["Created at"].max())].copy()
                    li_price = _money(li["Lineitem price"]) if "Lineitem price" in li.columns else pd.Series(dtype=float)
                    li_qty = pd.to_numeric(li["Lineitem quantity"], errors="coerce") if "Lineitem quantity" in li.columns else pd.Series(dtype=float)
                    li_disc = _money(li["Lineitem discount"]) if "Lineitem discount" in li.columns else pd.Series(0.0, index=li.index)
                    li["_line_net"] = (li_price * li_qty) - li_disc
                    if "Name" in li.columns:
                        per_order = li.groupby("Name")["_line_net"].sum()
                        val = float(per_order.dropna().sum())
                    else:
                        val = float(li["_line_net"].dropna().sum())
                out["value"] = val
                out["method"] = "line_items"

            # Alternatives for validation/debug
            try:
                if "Subtotal" in frame_orders.columns:
                    sub = _money(frame_orders["Subtotal"]).sum(skipna=True)
                    disc = _money(frame_orders["Total Discount"]).sum(skipna=True) if "Total Discount" in frame_orders.columns else 0.0
                    out["alt"]["subtotal_minus_discount"] = float(sub - disc)
            except Exception:
                pass
            try:
                if "Total" in frame_orders.columns:
                    tot = _money(frame_orders["Total"]).sum(skipna=True)
                    ship = _money(frame_orders["Shipping"]).sum(skipna=True) if "Shipping" in frame_orders.columns else 0.0
                    tax = _money(frame_orders["Taxes"]).sum(skipna=True) if "Taxes" in frame_orders.columns else 0.0
                    out["alt"]["total_minus_shipping_taxes"] = float(tot - ship - tax)
            except Exception:
                pass
            # Line-items aggregate as explicit alternative if canonical wasn't line_items
            try:
                if out.get("method") != "line_items" and all(c in d_kpi.columns for c in ["Lineitem price", "Lineitem quantity"]):
                    if "Name" in frame_orders.columns and "Name" in d_kpi.columns:
                        order_names = set(frame_orders["Name"].astype(str))
                        li = d_kpi[d_kpi["Name"].astype(str).isin(order_names)].copy()
                    else:
                        li = d_kpi[(d_kpi["Created at"] >= frame_orders["Created at"].min()) & (d_kpi["Created at"] <= frame_orders["Created at"].max())].copy()
                    li_price = _money(li["Lineitem price"]) if "Lineitem price" in li.columns else pd.Series(dtype=float)
                    li_qty = pd.to_numeric(li["Lineitem quantity"], errors="coerce") if "Lineitem quantity" in li.columns else pd.Series(dtype=float)
                    li_disc = _money(li["Lineitem discount"]) if "Lineitem discount" in li.columns else pd.Series(0.0, index=li.index)
                    li["_line_net"] = (li_price * li_qty) - li_disc
                    if "Name" in li.columns:
                        per_order = li.groupby("Name")["_line_net"].sum()
                        out["alt"]["line_items_aggregate"] = float(per_order.dropna().sum())
                    else:
                        out["alt"]["line_items_aggregate"] = float(li["_line_net"].dropna().sum())
            except Exception:
                pass
            return out

        # ACTUAL net sales - what really happened
        ns1_dbg = _netsales_debug(rec)
        ns0_dbg = _netsales_debug(pri)
        ns1_actual = ns1_dbg.get("value", float("nan"))
        ns0_actual = ns0_dbg.get("value", float("nan"))

        # Use ACTUAL values for all calculations
        o1 = o1_actual
        o0 = o0_actual
        ns1 = ns1_actual
        ns0 = ns0_actual

        # Calculate seasonal expectations SEPARATELY (for reference only)
        seasonal_expected = {}
        if seasonally_adjust and seasonal_data:
            try:
                def _sum_range(ts, start, end):
                    if ts is None: return None
                    return float(ts.loc[start.normalize():end.normalize()].sum())
                
                if seasonal_data.get('orders_ts') is not None:
                    o1_expected = int(round(max(0.0, _sum_range(seasonal_data['orders_ts'], rs, re))))
                    o0_expected = int(round(max(0.0, _sum_range(seasonal_data['orders_ts'], ps, pe))))
                    seasonal_expected['orders_recent'] = o1_expected
                    seasonal_expected['orders_prior'] = o0_expected
                    # Calculate what the model thinks the lift should be
                    if o0_expected > 0:
                        seasonal_expected['orders_expected_lift'] = (o1_expected - o0_expected) / o0_expected
                
                if seasonal_data.get('netsales_ts') is not None:
                    ns1_expected = max(0.0, _sum_range(seasonal_data['netsales_ts'], rs, re))
                    ns0_expected = max(0.0, _sum_range(seasonal_data['netsales_ts'], ps, pe))
                    seasonal_expected['net_sales_recent'] = ns1_expected
                    seasonal_expected['net_sales_prior'] = ns0_expected
                    if ns0_expected > 0:
                        seasonal_expected['net_sales_expected_lift'] = (ns1_expected - ns0_expected) / ns0_expected
                
                # Calculate "surprise" factor - how much actual differs from expected
                if o1_expected and o1_expected > 0:
                    seasonal_expected['orders_surprise'] = (o1 - o1_expected) / o1_expected
                if ns1_expected and ns1_expected > 0:
                    seasonal_expected['net_sales_surprise'] = (ns1 - ns1_expected) / ns1_expected
                    
            except Exception as e:
                # Silently fall back if seasonal calc fails
                seasonal_expected = {'error': str(e)}

        # AOV based on ACTUAL values
        aov1 = float(ns1/o1) if (o1 and not np.isnan(ns1)) else None
        aov0 = float(ns0/o0) if (o0 and not np.isnan(ns0)) else None

        # discount rate: total discount / subtotal (when available)
        def _disc_rate(frame_orders: pd.DataFrame) -> Optional[float]:
            if "Subtotal" in frame_orders.columns:
                sub = _money(frame_orders["Subtotal"]).sum(skipna=True)
                disc = _money(frame_orders["Total Discount"]).sum(skipna=True) if "Total Discount" in frame_orders.columns else 0.0
                if not np.isnan(sub) and sub>0:
                    return float(disc/(sub+1e-9))
            return None

        dr1 = _disc_rate(rec)
        dr0 = _disc_rate(pri)

        # Customer metrics: repeat within window and returning before window
        def _customer_metrics(frame_orders: pd.DataFrame, window_start: pd.Timestamp) -> Tuple[Optional[float], Optional[float], int]:
            keys = _identity_key(frame_orders).dropna()
            if keys.empty:
                return None, None, 0
            # Unique identified customers in this window
            unique = keys.dropna().unique()
            total = int(len(unique))
            if total < int(min_identified or 0):
                return None, None, total
            # Repeat purchase rate within window: customers with 2+ orders in this window
            counts = keys.value_counts()
            repeat_customers = int((counts > 1).sum())
            repeat_rate = float(repeat_customers / len(counts)) if len(counts) > 0 else None
            # Returning customer rate: had any order before the window start
            returning = int(sum((first_seen.get(k, window_start) < window_start) for k in unique))
            returning_rate = float(returning / total) if total > 0 else None
            return repeat_rate, returning_rate, total

        rep1, ret1, id1 = _customer_metrics(rec, rs)
        rep0, ret0, id0 = _customer_metrics(pri, ps)
        # New explicit metrics
        rrw1 = rep1  # repeat rate within window
        rrw0 = rep0
        rcs1 = ret1  # returning customer share (pre-window history)
        rcs0 = ret0
        ncr1 = (1.0 - rcs1) if (rcs1 is not None) else None
        ncr0 = (1.0 - rcs0) if (rcs0 is not None) else None

        # deltas based on ACTUAL values
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
            # New metrics
            "repeat_rate_within_window": _rel_delta(rrw1, rrw0),
            "returning_customer_share":  _rel_delta(rcs1, rcs0),
            "new_customer_rate":         _rel_delta(ncr1, ncr0),
        }

        # significance testing on ACTUAL values
        p = {
            "aov": None, "discount_rate": None,
            "repeat_rate_within_window": None,
            "returning_customer_share": None,
            "new_customer_rate": None,
        }
        
        # AOV: Welch t on per-order net values
        a1 = rec["_order_net"].dropna().values if "_order_net" in rec.columns else np.array([])
        a0 = pri["_order_net"].dropna().values if "_order_net" in pri.columns else np.array([])

        def _extract_p(val):
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

        # Discount rate: Welch t-test on per-order discount rates
        # Aligns test with the metric reported (average discount depth), not just share of discounted orders.
        def _per_order_discount_rates(frame_orders: pd.DataFrame) -> np.ndarray:
            if ("Total Discount" not in frame_orders.columns) or ("Subtotal" not in frame_orders.columns):
                return np.array([])
            sub = _money(frame_orders["Subtotal"]).astype(float)
            disc = _money(frame_orders["Total Discount"]).astype(float)
            valid = sub.notna() & disc.notna() & (sub > 0)
            if not valid.any():
                return np.array([])
            rates = (disc[valid] / sub[valid]).clip(lower=0.0, upper=1.0)
            return rates.astype(float).values

        try:
            r1 = _per_order_discount_rates(rec)
            r0 = _per_order_discount_rates(pri)
            if (r1.size > 1) and (r0.size > 1):
                mt_dr = welch_t_test(r1, r0)
                p_dr = _extract_p(mt_dr)
                p["discount_rate"] = p_dr
        except Exception:
            pass

        # Repeat rate: two-proportion on identified customers within window
        if rep1 is not None and rep0 is not None and id1>=min_identified and id0>=min_identified:
            x1 = int(round(rep1*id1)); n1 = id1
            x0 = int(round(rep0*id0)); n0 = id0
            pr_rep = two_proportion_test(x1,n1,x0,n0)
            pval_rep = float(pr_rep.p_value)
            p["repeat_rate_within_window"] = pval_rep
        # Returning rate: two-proportion on identified customers
        if ret1 is not None and ret0 is not None and id1>=min_identified and id0>=min_identified:
            x1r = int(round(ret1*id1)); n1r = id1
            x0r = int(round(ret0*id0)); n0r = id0
            pr_ret = two_proportion_test(x1r,n1r,x0r,n0r)
            pval_ret = float(pr_ret.p_value)
            p["returning_customer_share"] = pval_ret
            # new_customer_rate is 1 - returning; mirror p-value
            p["new_customer_rate"] = pval_ret

        # FDR on available p's
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

        # Build result body
        result = {
            "net_sales": None if np.isnan(ns1) else float(ns1),
            "orders": o1,
            "aov": aov1,
            "discount_rate": dr1,
            # New explicit metrics
            "repeat_rate_within_window": rrw1,
            "returning_customer_share": rcs1,
            "new_customer_rate": ncr1,
            "prior": {
                "net_sales": None if np.isnan(ns0) else float(ns0),
                "orders": o0,
                "aov": aov0,
                "discount_rate": dr0,
                # New explicit metrics
                "repeat_rate_within_window": rrw0,
                "returning_customer_share": rcs0,
                "new_customer_rate": ncr0,
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
        # Record method used + debug comparisons for transparency
        try:
            def _consistency(meta_key_prefix: str, dbg: dict):
                result.setdefault("meta", {})[f"{meta_key_prefix}_netsales_method"] = dbg.get("method")
                # compute diffs vs alternatives
                val = dbg.get("value")
                alts = dbg.get("alt", {}) or {}
                diffs = {}
                for k, v in alts.items():
                    try:
                        if v is None or val is None or np.isnan(val):
                            continue
                        if val == 0:
                            continue
                        diffs[k] = float((val - float(v)) / abs(val))
                    except Exception:
                        continue
                result["meta"][f"{meta_key_prefix}_netsales_alt_diffs"] = diffs
                # flag if any alternative differs >10%
                if any(abs(x) > 0.10 for x in diffs.values()):
                    result["meta"][f"{meta_key_prefix}_netsales_consistency_flag"] = True
            _consistency("recent", ns1_dbg)
            _consistency("prior", ns0_dbg)
        except Exception:
            # keep snapshot robust even if debug calc fails
            pass
        
        # Add seasonal expectations as metadata (never overrides actual)
        if seasonal_expected:
            result["seasonal_expected"] = seasonal_expected

        return result

    out = {"anchor": anchor, "L7": _summarize(7), "L28": _summarize(28)}
    out["meta"] = {
        "seasonal_adjusted": bool(seasonally_adjust),
        "seasonal_period": int(seasonal_period) if seasonally_adjust else None,
        "seasonal_method": seasonal_method if seasonally_adjust else None,
        "metric_version": "v2_repeat_metrics",
    }
    
    # convenience: top-level recent values for backward compatibility
    for label in ("L7","L28"):
        for k in (
            "net_sales","orders","aov","discount_rate",
            # New explicit metrics
            "repeat_rate_within_window","returning_customer_share","new_customer_rate",
            # Back-compat aliases
            "repeat_share","repeat_rate","returning_rate"
        ):
            out[label][k] = out[label].get(k)
    
    # include direction rule hint for UI
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
