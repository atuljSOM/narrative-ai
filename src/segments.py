
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List

def build_segments(g: pd.DataFrame, gross_margin: float, out_dir: str) -> List[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outputs = []
    last_by_cust = g.sort_values("Created at").groupby("customer_id").tail(1)

    winback = last_by_cust[(last_by_cust["days_since_last"]>=21) & (last_by_cust["days_since_last"]<=45)]
    winback = winback[["customer_id"]].drop_duplicates(); winback["segment"]="winback_21_45"
    winback["segment_n"]=len(winback); winback["baseline_rate"]=g["is_repeat"].mean(); winback["gross_margin"]=gross_margin
    p = Path(out_dir)/"segment_winback_21_45.csv"; winback.to_csv(p, index=False); outputs.append(str(p))

    freq = g.groupby("customer_id")["Name"].nunique().rename("orders")
    dormant = last_by_cust.join(freq, on="customer_id")
    dormant = dormant[(dormant["days_since_last"]>=60)&(dormant["days_since_last"]<=120)&(dormant["orders"]>=2)]
    dormant = dormant[["customer_id"]].drop_duplicates(); dormant["segment"]="dormant_multibuyers_60_120"
    dormant["segment_n"]=len(dormant); dormant["baseline_rate"]=g["is_repeat"].mean(); dormant["gross_margin"]=gross_margin
    p = Path(out_dir)/"segment_dormant_multibuyers_60_120.csv"; dormant.to_csv(p, index=False); outputs.append(str(p))

    top_sku = g.groupby("lineitem_any")["net_sales"].sum().sort_values(ascending=False).head(1)
    sku_name = top_sku.index[0] if len(top_sku)>0 else "unknown"
    top_buyers = g[g["lineitem_any"]==sku_name][["customer_id"]].drop_duplicates()
    top_buyers["segment"]="bestseller_amplify"; top_buyers["segment_n"]=len(top_buyers)
    top_buyers["baseline_rate"]=g["units_per_order"].mean(); top_buyers["gross_margin"]=gross_margin
    p = Path(out_dir)/"segment_bestseller_amplify.csv"; top_buyers.to_csv(p, index=False); outputs.append(str(p))

    cust_disc = g.groupby("customer_id")["discount_rate"].mean().sort_values(ascending=False)
    high_disc = cust_disc[cust_disc>=0.20].index
    disc_df = pd.DataFrame({"customer_id": high_disc}); disc_df["segment"]="discount_hygiene"
    disc_df["segment_n"]=len(disc_df); disc_df["baseline_rate"]=g["discount_rate"].mean(); disc_df["gross_margin"]=gross_margin
    p = Path(out_dir)/"segment_discount_hygiene.csv"; disc_df.to_csv(p, index=False); outputs.append(str(p))

    # bundle
    import zipfile
    bundle = Path(out_dir)/"segments_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in outputs: zf.write(f, Path(f).name)
    outputs.append(str(bundle))
    return outputs
