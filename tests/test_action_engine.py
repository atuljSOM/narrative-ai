
import pandas as pd, numpy as np
from pathlib import Path
from src.action_engine import select_actions
from src.features import aligned_periods_summary
from src.utils import get_config

def make_synth(n=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-06-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Name": [f"#{i}" for i in range(n)],
        "customer_id": [f"user{i%120}@x.com" for i in range(n)],
        "Created at": dates,
        "net_sales": rng.normal(80, 10, size=n).clip(20, 200),
        "discount_rate": rng.uniform(0.0, 0.2, size=n),
        "units_per_order": rng.integers(1, 3, size=n),
        "lineitem_any": rng.choice(["SKU_A","SKU_B","SKU_C"], size=n, p=[0.5,0.3,0.2]),
        "AOV": rng.normal(80, 10, size=n).clip(20, 200),
        "is_repeat": (dates > dates.min() + pd.Timedelta(days=30)).astype(int),
        "days_since_last": rng.integers(10, 130, size=n),
    })
    return df

def test_action_or_pilot(tmp_path):
    g = make_synth()
    aligned = aligned_periods_summary(g, min_window_n=100)
    cfg = get_config()
    plays = str(Path(__file__).resolve().parents[1]/"templates"/"playbooks.yml")
    receipts = tmp_path/"receipts"; receipts.mkdir(parents=True, exist_ok=True)
    out = select_actions(g, aligned, cfg, plays, str(receipts))
    assert "backlog" in out
    assert (len(out["actions"])>0) or (len(out.get("pilot_actions",[]))>0)
