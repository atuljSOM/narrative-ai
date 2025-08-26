
import pandas as pd
from src.features import compute_features, aligned_periods_summary

def test_features_and_aligned_periods():
    df = pd.DataFrame({
        "Name": [f"#{i}" for i in range(1, 41)],
        "customer_id": [f"a@x.com"]*20 + [f"b@x.com"]*20,
        "Created at": pd.date_range("2025-07-10", periods=40, freq="D"),
        "net_sales": [100.0]*40,
        "discount_rate": [0.1]*40,
        "units_per_order": [1]*40,
        "Lineitem name": ["SKU"]*40,
        "Currency": ["USD"]*40,
    })
    g = compute_features(df.rename(columns={"Created at":"Created at"}))
    aligned = aligned_periods_summary(g, min_window_n=30)
    assert aligned["recent_n"] >= 0
    assert aligned["window_days"] in (7,28,56)
