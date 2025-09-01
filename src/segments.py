
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

    # Subscription nudge: customers with ≥3 orders of the same product in 90 days
    try:
        maxd = pd.to_datetime(g["Created at"]).max()
        start90 = maxd - pd.Timedelta(days=90)
        gg = g[g["Created at"] >= start90].copy()
        if "lineitem_any" in gg.columns:
            rep = (
                gg.groupby(["customer_id", "lineitem_any"])['Name']
                  .nunique()
                  .reset_index(name='orders_product')
            )
            cohort = rep[rep['orders_product'] >= 3]
            sub_seg = cohort[["customer_id"]].drop_duplicates()
            sub_seg["segment"] = "subscription_nudge"
            p = Path(out_dir)/"segment_subscription_nudge.csv"; sub_seg.to_csv(p, index=False); outputs.append(str(p))
    except Exception:
        pass

    # Ingredient education: first-time technical buyers
    try:
        freq_all = g.groupby("customer_id")["Name"].nunique().rename("orders_total")
        last_win = g.sort_values("Created at").merge(freq_all, left_on="customer_id", right_index=True, how="left")
        tech = last_win["lineitem_any"].astype(str).str.lower().str.contains(r"retinol|acid|aha|bha|salicy|glycolic|lactic|peptide|niacinamide|vitamin c|ascorb", regex=True)
        ft = last_win["orders_total"].fillna(0).astype(int).eq(1)
        edu = last_win[ft & tech][["customer_id"]].drop_duplicates(); edu["segment"] = "ingredient_education"
        if not edu.empty:
            p = Path(out_dir)/"segment_ingredient_education.csv"; edu.to_csv(p, index=False); outputs.append(str(p))
    except Exception:
        pass

    # Empty bottle: near depletion based on parsed size
    try:
        last = g.sort_values("Created at").groupby("customer_id").tail(1).copy()
        if "lineitem_any" in last.columns and "days_since_last" in last.columns:
            names = last["lineitem_any"].astype(str).str.lower()
            size_days = []
            for s in names:
                if "100ml" in s or "3.4 oz" in s or "3.4oz" in s:
                    size_days.append(75)
                elif "50ml" in s or "1.7 oz" in s or "1.7oz" in s:
                    size_days.append(40)
                elif "30ml" in s or "1 oz" in s or "1oz" in s:
                    size_days.append(25)
                else:
                    size_days.append(None)
            last = last.assign(_deplete_days=size_days)
            window = last[~pd.isna(last["_deplete_days"])].copy()
            m = (window["days_since_last"] >= (window["_deplete_days"] - 3)) & (window["days_since_last"] <= (window["_deplete_days"] + 3))
            eb = window.loc[m, ["customer_id"]].drop_duplicates(); eb["segment"] = "empty_bottle"
            if not eb.empty:
                p = Path(out_dir)/"segment_empty_bottle.csv"; eb.to_csv(p, index=False); outputs.append(str(p))
    except Exception:
        pass

    # Sample to full-size: sample/travel buyers 14–21 days ago without full-size since
    try:
        anchor = pd.to_datetime(g["Created at"]).max()
        win_start = anchor - pd.Timedelta(days=21)
        win_end   = anchor - pd.Timedelta(days=14)
        gg2 = g[(g["Created at"] >= win_start) & (g["Created at"] <= win_end)].copy()
        if "lineitem_any" in gg2.columns:
            li = gg2["lineitem_any"].astype(str).str.lower()
            sample_mask = li.str.contains(r"sample|travel|mini|trial", regex=True)
            sample_orders = gg2[sample_mask]
            if not sample_orders.empty:
                sample_custs = set(sample_orders["customer_id"].astype(str))
                after = g[g["Created at"] > win_end].copy()
                is_full = (~after["lineitem_any"].astype(str).str.lower().str.contains(r"sample|travel|mini|trial", regex=True))
                full_buys = set(after[is_full]["customer_id"].astype(str))
                targets = pd.DataFrame({"customer_id": list(sample_custs.difference(full_buys))})
                if not targets.empty:
                    targets["segment"] = "sample_to_full"
                    p = Path(out_dir)/"segment_sample_to_full.csv"; targets.to_csv(p, index=False); outputs.append(str(p))
    except Exception:
        pass

    # Routine builder: skincare single-product purchasers in recent 60d
    try:
        anchor2 = pd.to_datetime(g["Created at"]).max()
        recent_start = anchor2 - pd.Timedelta(days=60)
        lookback_start = anchor2 - pd.Timedelta(days=90)
        gr = g[(g["Created at"] >= recent_start)].copy()
        if "category" in gr.columns:
            gr_skin = gr[gr["category"].astype(str).str.lower() == "skincare"].copy()
        else:
            gr_skin = gr.copy()
        cand_ids = set(gr_skin["customer_id"].astype(str))
        if cand_ids:
            gl = g[(g["Created at"] >= lookback_start)].copy()
            gl["customer_id"] = gl["customer_id"].astype(str)
            if "lineitem_any" in gl.columns:
                k = gl.groupby("customer_id")["lineitem_any"].nunique()
                single_prod_ids = set(k[k <= 1].index)
            else:
                single_prod_ids = set()
            targets = list(cand_ids.intersection(single_prod_ids))
            if targets:
                df_rb = pd.DataFrame({"customer_id": targets}); df_rb["segment"] = "routine_builder"
                p = Path(out_dir)/"segment_routine_builder.csv"; df_rb.to_csv(p, index=False); outputs.append(str(p))
    except Exception:
        pass

    # bundle
    import zipfile
    bundle = Path(out_dir)/"segments_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in outputs: zf.write(f, Path(f).name)
    outputs.append(str(bundle))
    return outputs
