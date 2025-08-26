
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

DEFAULTS = {
    "MIN_N_WINBACK": 150,
    "MIN_N_SKU": 60,
    "AOV_EFFECT_FLOOR": 0.03,
    "REPEAT_PTS_FLOOR": 0.02,
    "DISCOUNT_PTS_FLOOR": 0.03,
    "FDR_ALPHA": 0.10,
    "FINANCIAL_FLOOR": 300.0,
    "GROSS_MARGIN": 0.70,
    "EFFORT_BUDGET": 8,
    "PILOT_BUDGET_CAP": 200.0,
    "PILOT_AUDIENCE_FRACTION": 0.2,
}

def load_env(path: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    p = Path(path or ".env")
    if p.exists():
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip()
    return env

def _f(env: Dict[str,str], k: str, default: float) -> float:
    try: return float(env.get(k, default))
    except: return float(default)

def _i(env: Dict[str,str], k: str, default: int) -> int:
    try: return int(env.get(k, default))
    except: return int(default)

def get_config(env: Optional[Dict[str,str]] = None) -> Dict[str, Any]:
    env = env or load_env()
    return {
        "MIN_N_WINBACK": _i(env,"MIN_N_WINBACK", DEFAULTS["MIN_N_WINBACK"]),
        "MIN_N_SKU": _i(env,"MIN_N_SKU", DEFAULTS["MIN_N_SKU"]),
        "AOV_EFFECT_FLOOR": _f(env,"AOV_EFFECT_FLOOR", DEFAULTS["AOV_EFFECT_FLOOR"]),
        "REPEAT_PTS_FLOOR": _f(env,"REPEAT_PTS_FLOOR", DEFAULTS["REPEAT_PTS_FLOOR"]),
        "DISCOUNT_PTS_FLOOR": _f(env,"DISCOUNT_PTS_FLOOR", DEFAULTS["DISCOUNT_PTS_FLOOR"]),
        "FDR_ALPHA": _f(env,"FDR_ALPHA", DEFAULTS["FDR_ALPHA"]),
        "FINANCIAL_FLOOR": _f(env,"FINANCIAL_FLOOR", DEFAULTS["FINANCIAL_FLOOR"]),
        "GROSS_MARGIN": _f(env,"GROSS_MARGIN", DEFAULTS["GROSS_MARGIN"]),
        "EFFORT_BUDGET": _i(env,"EFFORT_BUDGET", DEFAULTS["EFFORT_BUDGET"]),
        "PILOT_BUDGET_CAP": _f(env,"PILOT_BUDGET_CAP", DEFAULTS["PILOT_BUDGET_CAP"]),
        "PILOT_AUDIENCE_FRACTION": _f(env,"PILOT_AUDIENCE_FRACTION", DEFAULTS["PILOT_AUDIENCE_FRACTION"]),
    }

def winsorize_series(s: pd.Series, lower_p=0.01, upper_p=0.99) -> pd.Series:
    if s.dropna().empty: return s
    lo = s.quantile(lower_p); hi = s.quantile(upper_p)
    return s.clip(lower=lo, upper=hi)

def aligned_windows(max_date, n_recent: int):
    import pandas as pd
    end_recent = pd.Timestamp(max_date.normalize())
    start_recent = end_recent - pd.Timedelta(days=n_recent-1)
    end_prior = start_recent - pd.Timedelta(days=1)
    start_prior = end_prior - pd.Timedelta(days=n_recent-1)
    return start_recent, end_recent, start_prior, end_prior

def safe_make_dirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

def read_json(path: str):
    p = Path(path)
    if not p.exists(): return None
    return json.loads(p.read_text())
