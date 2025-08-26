
import pandas as pd
from pathlib import Path
from src.segments import build_segments

def test_segments_bundle(tmp_path):
    g = pd.DataFrame({
        "Name": [f"#{i}" for i in range(1,61)],
        "customer_id": [f"user{i%15}@x.com" for i in range(1,61)],
        "Created at": pd.date_range("2025-06-01", periods=60, freq="D"),
        "net_sales": [50.0]*60,
        "discount_rate": [0.1 if i%3 else 0.4 for i in range(60)],
        "units_per_order": [1]*60,
        "lineitem_any": ["SKU_A"]*30 + ["SKU_B"]*30,
        "AOV": [50.0]*60,
        "is_repeat": [1 if i%2 else 0 for i in range(60)],
        "days_since_last": [20 + (i%80) for i in range(60)],
    })
    out_dir = tmp_path/"segments"; files = build_segments(g, 0.7, str(out_dir))
    assert sum(1 for f in files if f.endswith(".csv")) == 4
    assert any(f.endswith(".zip") for f in files)
    df = pd.read_csv([f for f in files if f.endswith("segment_winback_21_45.csv")][0])
    assert "segment_n" in df.columns
