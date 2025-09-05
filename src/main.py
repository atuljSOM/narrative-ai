# src/main.py
from __future__ import annotations
import argparse
from pathlib import Path
from os.path import relpath
import shutil

from .utils import (
    get_config, safe_make_dirs, write_json,
    choose_window, financial_floor,
    kpi_snapshot_with_deltas,
    identity_coverage,
)
from .utils import normalize_product_name
import pandas as pd
from .load import (
    load_csv, load_inventory_csv, compute_inventory_metrics,
    load_orders_csv, load_order_items_csv, preprocess
)
from .features import compute_features, aligned_periods_summary
from .contracts import IngestionContract, FeatureContract
from .segments import build_segments
from .charts import generate_charts
from .briefing import render_briefing
from .action_engine import (
    select_actions,
    write_actions_log,
    build_receipts,
    evidence_for_action,
    track_implementation_status,
    track_action_results,
)
from .action_engine import ActionTracker
from .copykit import render_copy_for_actions
from .validation import DataValidationEngine  # NEW IMPORT


def debug_dataframe_consistency(df, g, aligned, df_for_charts=None, receipts_dir="receipts"):
    """Debug dataframe inconsistencies that cause erroneous results"""
    import json as _json
    from pathlib import Path as _Path
    import pandas as _pd
    debug_report = {}

    print("🔍 DATAFRAME CONSISTENCY DEBUG")
    print("=" * 50)

    # 1. Basic shape comparison
    debug_report['shapes'] = {
        'df_raw_rows': len(df),
        'g_features_rows': len(g),
        'df_charts_rows': len(df_for_charts) if df_for_charts is not None else None
    }
    print(f"📊 Shapes: df={len(df)}, g={len(g)}, charts={len(df_for_charts) if df_for_charts is not None else 'None'}")

    # 2. Date range comparison
    try:
        df_dates = _pd.to_datetime(df['Created at'], errors='coerce')
        g_dates = _pd.to_datetime(g['Created at'], errors='coerce') if 'Created at' in g.columns else _pd.Series([], dtype='datetime64[ns]')

        debug_report['date_ranges'] = {
            'df_min': df_dates.min(),
            'df_max': df_dates.max(),
            'g_min': g_dates.min() if len(g_dates) else None,
            'g_max': g_dates.max() if len(g_dates) else None,
            'anchor': aligned.get('anchor')
        }

        print(f"📅 Date ranges:")
        print(f"   df: {df_dates.min()} to {df_dates.max()}")
        print(f"   g:  {g_dates.min() if len(g_dates) else None} to {g_dates.max() if len(g_dates) else None}")
        print(f"   anchor: {aligned.get('anchor')}")

        # Check for date misalignment
        try:
            if len(g_dates) and abs((df_dates.max() - g_dates.max()).days) > 0:
                print("⚠️  DATE MISMATCH between df and g!")
        except Exception:
            pass
    except Exception as e:
        print(f"❌ Date comparison failed: {e}")

    # 3. Customer identity comparison
    try:
        from .utils import standardize_customer_key

        df_customers = standardize_customer_key(df).dropna().unique()
        g_customers = g['customer_id'].dropna().unique() if 'customer_id' in g.columns else []

        debug_report['customers'] = {
            'df_unique': int(len(df_customers)),
            'g_unique': int(len(g_customers)),
            'overlap': int(len(set(df_customers) & set(g_customers))) if len(g_customers) > 0 else 0
        }

        print(f"👥 Customer counts: df={len(df_customers)}, g={len(g_customers)}")
        if len(g_customers) > 0 and len(df_customers) > 0:
            overlap = len(set(df_customers) & set(g_customers))
            print(f"   Overlap: {overlap} ({overlap/len(df_customers)*100:.1f}%)")

            if overlap < len(df_customers) * 0.9:
                print("⚠️  CUSTOMER MISMATCH: <90% overlap between df and g!")

    except Exception as e:
        print(f"❌ Customer comparison failed: {e}")

    # 4. Revenue calculation comparison
    try:
        def _money(s):
            return _pd.to_numeric(s, errors='coerce')

        # Method 1: Subtotal - Discount (order level)
        if all(c in df.columns for c in ['Subtotal', 'Total Discount', 'Name']):
            df_dedup = df.drop_duplicates(subset=['Name'])
            rev1 = (_money(df_dedup['Subtotal']) - _money(df_dedup['Total Discount'])).sum()
        else:
            rev1 = None

        # Method 2: From g features
        rev2 = g['net_sales'].sum() if 'net_sales' in g.columns else None

        # Method 3: L28 from aligned
        rev3 = (aligned.get('L28', {}) or {}).get('net_sales')

        debug_report['revenue_methods'] = {
            'subtotal_minus_discount': float(rev1) if rev1 is not None else None,
            'g_net_sales': float(rev2) if rev2 is not None else None,
            'aligned_l28': float(rev3) if rev3 is not None else None
        }

        print("💰 Revenue comparison:")
        print(f"   Subtotal-Discount: ${rev1:,.0f}" if rev1 is not None else "   Subtotal-Discount: None")
        print(f"   G net_sales: ${rev2:,.0f}" if rev2 is not None else "   G net_sales: None")
        print(f"   Aligned L28: ${rev3:,.0f}" if rev3 is not None else "   Aligned L28: None")

        # Check for major discrepancies
        revenues = [r for r in [rev1, rev2, rev3] if r is not None]
        if len(revenues) > 1:
            max_rev, min_rev = max(revenues), min(revenues)
            if max_rev > 0 and (max_rev - min_rev) / max_rev > 0.1:
                print("⚠️  REVENUE MISMATCH: >10% difference between calculation methods!")

    except Exception as e:
        print(f"❌ Revenue comparison failed: {e}")

    # 5. Order count comparison
    try:
        df_orders = df['Name'].nunique() if 'Name' in df.columns else len(df)
        g_orders = g['Name'].nunique() if 'Name' in g.columns else len(g)
        aligned_orders = (aligned.get('L28', {}) or {}).get('orders')

        debug_report['order_counts'] = {
            'df_unique_orders': int(df_orders),
            'g_unique_orders': int(g_orders),
            'aligned_l28_orders': int(aligned_orders) if aligned_orders is not None else None
        }

        print("📦 Order counts:")
        print(f"   df unique: {df_orders}")
        print(f"   g unique: {g_orders}")
        print(f"   aligned L28: {aligned_orders}")

        if abs(df_orders - g_orders) > df_orders * 0.05:
            print("⚠️  ORDER COUNT MISMATCH: >5% difference between df and g!")

    except Exception as e:
        print(f"❌ Order count comparison failed: {e}")

    # 6. Window alignment check
    try:
        anchor = aligned.get('anchor')
        if anchor:
            anchor = _pd.Timestamp(anchor)
            l28_window_days = (aligned.get('L28', {}) or {}).get('window_days', 28)

            # Expected L28 range
            l28_end = anchor.normalize() + _pd.Timedelta(hours=23, minutes=59, seconds=59)
            l28_start = l28_end.normalize() - _pd.Timedelta(days=l28_window_days - 1)

            # Count orders in expected L28 window
            df_in_window = df[
                (_pd.to_datetime(df['Created at'], errors='coerce') >= l28_start) &
                (_pd.to_datetime(df['Created at'], errors='coerce') <= l28_end)
            ]
            window_orders = df_in_window['Name'].nunique() if 'Name' in df_in_window.columns else len(df_in_window)

            debug_report['window_check'] = {
                'l28_start': str(l28_start),
                'l28_end': str(l28_end),
                'expected_orders': int(window_orders),
                'aligned_orders': int(aligned.get('L28', {}).get('orders')) if (aligned.get('L28', {}) or {}).get('orders') is not None else None
            }

            print(f"🎯 L28 Window: {l28_start.date()} to {l28_end.date()}")
            print(f"   Orders in window: {window_orders}")
            print(f"   Aligned reports: {aligned.get('L28', {}).get('orders')}")

            try:
                ao = (aligned.get('L28', {}) or {}).get('orders')
                if ao and abs(window_orders - ao) > max(window_orders, ao) * 0.1:
                    print("⚠️  WINDOW MISMATCH: Aligned L28 doesn't match expected window!")
            except Exception:
                pass

    except Exception as e:
        print(f"❌ Window check failed: {e}")

    # Save debug report
    debug_path = _Path(receipts_dir) / "dataframe_debug.json"
    with open(debug_path, 'w') as f:
        _json.dump(debug_report, f, indent=2, default=str)

    print(f"\n📋 Debug report saved to: {debug_path}")

    return debug_report


def run(csv_path: str, brand: str, out_dir: str, inventory_path: str | None = None, order_items_path: str | None = None) -> None:
    cfg = get_config()

    # --- output dirs
    safe_make_dirs(out_dir)
    receipts_dir = Path(out_dir) / "receipts"
    safe_make_dirs(str(receipts_dir))
    qa_path = str(receipts_dir / "qa_report.json")

    # --- load & features (flexible orders/items)
    items_df = None
    if order_items_path:
        orders_denorm, _, _ = load_orders_csv(csv_path, has_line_items=False)
        items_df = load_order_items_csv(order_items_path)
    else:
        orders_denorm, has_items, items_df = load_orders_csv(csv_path, has_line_items=None)
    # Preprocess orders for engine features
    df, qa = preprocess(orders_denorm)

    # Phase 1: Data contracts + shims (non-breaking)
    try:
        ic = IngestionContract(orders_df=orders_denorm, items_df=items_df)
        dq_contract = ic.data_quality()
    except Exception:
        dq_contract = {}

    # Optional enrichment behind feature flag
    try:
        if bool(cfg.get('FEATURES_DYNAMIC_PRODUCTS', False)):
            df_enriched, meta = FeatureContract.build_g_orders(df, items_df=items_df)
            # Only add additive fields to preserve existing behavior
            for col in ['primary_product','products_concat','products_concat_qty','products_struct','has_sample','has_supplement','category_mode_qty']:
                if col in df_enriched.columns:
                    df[col] = df_enriched[col]
            # Prefer category_mode_qty if available
            if 'category_mode_qty' in df.columns:
                df['category'] = df['category_mode_qty'].fillna(df['category'])
            # Debug sample of products_struct (write whenever dynamic products is enabled)
            try:
                if 'products_struct' in df.columns:
                    sample_df = df[['Name','products_struct']].copy()
                    # keep rows where struct is a non-empty list
                    def _non_empty(x):
                        try:
                            return isinstance(x, (list, tuple)) and len(x) > 0
                        except Exception:
                            return False
                    sample_df = sample_df[sample_df['products_struct'].apply(_non_empty)]
                    sample = sample_df.head(50).to_dict(orient='records')
                    write_json(str(Path(out_dir)/'receipts'/'products_struct_sample.json'), sample)
            except Exception as e:
                print(f"[warn] failed to write products_struct_sample.json: {e}")

            # Normalization debug: sample raw titles and parsed (base,size)
            try:
                titles = []
                # Prefer items_df product_title; else fall back to Lineitem name
                if items_df is not None and 'product_title' in items_df.columns:
                    tser = items_df['product_title'].astype(str).dropna().unique().tolist()
                    titles = tser[:200]
                elif 'Lineitem name' in df.columns:
                    tser = df['Lineitem name'].astype(str).dropna().unique().tolist()
                    titles = tser[:200]
                if titles:
                    dbg = []
                    for t in titles[:200]:
                        base, size = normalize_product_name(t)
                        dbg.append({'title': t, 'base': base, 'size': size})
                    write_json(str(Path(out_dir)/'receipts'/'normalization_debug.json'), dbg)
            except Exception as e:
                print(f"[warn] failed to write normalization_debug.json: {e}")
    except Exception as e:
        print(f"[warn] FeatureContract enrichment skipped: {e}")
    g = compute_features(df)

    # NOTE: We standardize on a single canonical orders frame (df) for KPIs and engine.
    # Features (g) are derived 1:1 from df and should not change the row set.

    # segments
    seg_dir = Path(out_dir) / "segments"
    seg_files = build_segments(g, cfg.get("GROSS_MARGIN", 0.70), str(seg_dir), cfg)

    # (charts generation moved to after actions are selected)

    # --- KPI snapshot with deltas (use canonical df)
    aligned_for_template = kpi_snapshot_with_deltas(
        df,
        seasonally_adjust=bool(cfg.get("SEASONAL_ADJUST", False)),
        seasonal_period=int(cfg.get("SEASONAL_PERIOD", 7)),
    )

    # --- adaptive knobs from snapshot
    l7_orders  = int(aligned_for_template.get("L7", {}).get("orders") or 0)
    l28_orders = int(aligned_for_template.get("L28", {}).get("orders") or 0)
    cfg["CHOSEN_WINDOW"] = choose_window(
        l7_orders=l7_orders, l28_orders=l28_orders,
        policy=str(cfg.get("WINDOW_POLICY", "auto")).lower(),
    )
    l28_net_sales = float(aligned_for_template.get("L28", {}).get("net_sales") or 0.0)
    if str(cfg.get("FINANCIAL_FLOOR_MODE", "auto")).lower() == "auto":
        cfg["FINANCIAL_FLOOR"] = float(
            financial_floor(l28_net_sales, float(cfg.get("GROSS_MARGIN", 0.70)))
        )

    # --- inventory (optional)
    inventory_df = None
    inventory_metrics = None
    if inventory_path:
        try:
            inventory_df = load_inventory_csv(inventory_path)
            inventory_metrics = compute_inventory_metrics(
                inventory_df, df,
                lead_time_days=int(cfg.get('INVENTORY_LEAD_TIME_DAYS', 14)),
                z=float(cfg.get('INVENTORY_SAFETY_Z', 1.64)),
                safety_floor=int(cfg.get('INVENTORY_SAFETY_STOCK', 0))
            )
        except Exception as e:
            print(f"[warn] inventory load/metrics failed: {e}")

    # --- actions
    plays = str(Path(Path(__file__).resolve().parent.parent) / "templates" / "playbooks.yml")
    # Pass the nested KPI snapshot directly; the engine normalizes internally
    actions = select_actions(g, aligned_for_template, cfg, plays, str(receipts_dir), inventory_metrics=inventory_metrics)

    receipts = build_receipts(aligned_for_template, actions)

    # --- Charts (new): generate with feature df and copy near the HTML ---
    chart_out_dir = Path(out_dir) / "charts"

    # Charts should read from the same canonical orders frame for consistency
    df_for_charts = df

    # Run a consistency debug snapshot before charts for easier diagnosis
    try:
        debug_dataframe_consistency(
            df=df,
            g=g,
            aligned=aligned_for_template,
            df_for_charts=df_for_charts,
            receipts_dir=str(receipts_dir)
        )
    except Exception as e:
        print(f"[warn] dataframe consistency debug failed: {e}")

    chart_data = generate_charts(
        g=g,
        aligned=aligned_for_template,
        actions=actions,
        out_dir=str(chart_out_dir),
        df=df_for_charts,
        chosen_window=str(cfg.get("CHOSEN_WINDOW", "L28")),
        charts_mode=str(cfg.get("CHARTS_MODE", "detailed")),
        inventory_metrics=inventory_metrics
    ) or {}

    # Write a small debug sample for product charts troubleshooting
    try:
        debug_sample = (df_for_charts.head(50) if isinstance(df_for_charts, pd.DataFrame) else pd.DataFrame())
        debug_sample.to_csv(Path(out_dir)/"receipts"/"df_for_charts_sample.csv", index=False)
        # Basic counts
        debug_counts = {
            'rows': int(len(df_for_charts)) if isinstance(df_for_charts, pd.DataFrame) else 0,
            'recent_30_rows': int(
                df_for_charts[df_for_charts['Created at'] >= pd.to_datetime(df_for_charts['Created at'], errors='coerce').max() - pd.Timedelta(days=30)].shape[0]
            ) if isinstance(df_for_charts, pd.DataFrame) and 'Created at' in df_for_charts.columns else 0,
            'has_customer_email': bool(isinstance(df_for_charts, pd.DataFrame) and 'Customer Email' in df_for_charts.columns),
            'has_customer_id': bool(isinstance(df_for_charts, pd.DataFrame) and 'customer_id' in df_for_charts.columns),
            'has_lineitem_name': bool(isinstance(df_for_charts, pd.DataFrame) and 'Lineitem name' in df_for_charts.columns),
        }
        write_json(str(Path(out_dir)/"receipts"/"df_for_charts_counts.json"), debug_counts)
    except Exception:
        pass

    briefing_dir = Path(out_dir) / "briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    charts_brief_dir = briefing_dir / "charts"
    charts_brief_dir.mkdir(parents=True, exist_ok=True)

    charts_map_rel: dict[str, str] = {}
    chart_paths_abs_resolved: list[str] = []
    for name, src in chart_data.items():
        try:
            src_path = Path(src)
            if not src_path.exists():
                print(f"[warn] chart missing on disk: {src}")
                continue
            dst_path = charts_brief_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            charts_map_rel[name] = str(dst_path.relative_to(briefing_dir))
            chart_paths_abs_resolved.append(str(src_path.resolve()))
        except Exception as e:
            print(f"[warn] failed to copy chart {src} -> {dst_path}: {e}")
    
    # --- DATA VALIDATION (NEW) ---
    validator = DataValidationEngine()
    validation_results = validator.run_all_checks(
        df=df,
        aligned=aligned_for_template,
        actions=actions.get("actions", []),
        qa=qa,
        inventory=inventory_df,
        inventory_metrics=inventory_metrics,
        config=cfg,
        orders_df=orders_denorm,
        items_df=items_df
    )

    # Gate downstream action lists on critical validation failures (e.g., AOV inconsistency)
    try:
        checks = (validation_results or {}).get('checks', {})
        aov_check = checks.get('AOV Consistency', {})
        overall = (validation_results or {}).get('overall_status')
        should_gate = (aov_check.get('status') == 'red') or (overall == 'red')
        if should_gate:
            reason = aov_check.get('message') or 'Critical data validation issues'
            # Demote actions to watchlist; annotate with blocking reason
            blocked = []
            for key in ['actions', 'pilot_actions']:
                lst = actions.get(key, []) or []
                for a in lst:
                    a.setdefault('notes', []).append(f"Blocked by validation: {reason}")
                    a['__blocked_by_validation__'] = True
                blocked.extend(lst)
                actions[key] = []
            actions['watchlist'] = (actions.get('watchlist', []) or []) + blocked
    except Exception:
        # Never fail the run due to gating logic; surfaces via validation report anyway
        pass
    
    # Save validation report
    validation_path = receipts_dir / "validation_report.json"
    write_json(str(validation_path), validation_results)
    
    # Generate HTML panel for briefing
    validation_html = validator.to_html_panel(validation_results)
    
    # Print validation summary to console
    print(f"\nData Validation: {validation_results['summary']}")
    if validation_results['critical_issues']:
        print("Critical Issues:")
        for issue in validation_results['critical_issues']:
            print(f"  - {issue}")
    if validation_results['warnings']:
        print("Warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")

    # copy assets for selected actions/pilots
    assets_dir = Path(out_dir) / "briefings" / "assets"
    selected_for_copy = (actions.get("actions", []) + actions.get("pilot_actions", []))
    for a in selected_for_copy:
        a["brand"] = brand
    copy_assets = render_copy_for_actions(
        str(Path(Path(__file__).resolve().parent.parent) / "templates"),
        str(assets_dir),
        selected_for_copy,
    )

    # --- Performance tracking summary for template (if any)
    tracker = ActionTracker(str(receipts_dir))
    performance_summary = tracker.get_performance_summary()
    pending_actions = tracker.get_pending_actions()

    # receipts summary file
    summary_path = receipts_dir / "run_summary.json"
    # Data quality snapshot (Phase 0)
    dq_orders = identity_coverage(df)
    has_line_items = bool(items_df is not None and not getattr(items_df, 'empty', True))
    has_sku = bool(has_line_items and any(c in items_df.columns for c in ['sku','variant_id','product_id']))
    product_coverage = 0.0
    try:
        if 'Lineitem name' in orders_denorm.columns:
            s = orders_denorm['Lineitem name'].astype(str).str.strip()
            product_coverage = float((s != '').mean())
    except Exception:
        pass

    write_json(str(summary_path), {
        # Use KPI snapshot (customer-based) as primary aligned for run_summary
        "aligned": aligned_for_template,
        # Keep same aligned structure for engine/briefing reference
        "aligned_order": aligned_for_template,
        "data_quality": {
            **dq_orders,
            **dq_contract,  # contract’s assessment (superset, safe to merge)
            "has_line_items": has_line_items,
            "has_sku": has_sku,
            "product_coverage": product_coverage,
        },
        "charts_abs": chart_paths_abs_resolved,
        "charts_rel": list(charts_map_rel.values()),
        "charts_map": charts_map_rel,
        "segments": seg_files,
        "actions": actions.get("actions", []),
        "watchlist": actions.get("watchlist", []),
        "pilot_actions": actions.get("pilot_actions", []),
        "backlog": actions.get("backlog", []),
        "performance_summary": performance_summary,
        "pending_actions": pending_actions,
    })
    write_actions_log(str(receipts_dir), actions.get("actions", []))

    # render briefing
    outputs = {
        "charts": list(charts_map_rel.values()),  # backward-compat (not used by new template)
        "charts_map": charts_map_rel,
        "segments_bundle": [s for s in seg_files if s.endswith(".zip")][0] if seg_files else "",
        "actions": actions.get("actions", []),
        "watchlist": actions.get("watchlist", []),
        "pilot_actions": actions.get("pilot_actions", []),
        "backlog": actions.get("backlog", []),
        "confidence_mode": actions.get("confidence_mode"),
        "cfg": cfg,
        "copy_assets": copy_assets,
        "receipts": receipts,
        "validation_html": validation_html,  # Pass to template
        "validation_results": validation_results,  # Pass full results too
        "performance_summary": performance_summary,
        "pending_actions": pending_actions,
        "inventory": inventory_df.to_dict(orient='records') if inventory_df is not None else None,
        "inventory_metrics": inventory_metrics.to_dict(orient='records') if inventory_metrics is not None else None,
    }

    # Build a concise inventory summary for the briefing
    if inventory_metrics is not None:
        try:
            mm = inventory_metrics.copy()
            default_cover = int(float(((cfg.get('INVENTORY_MIN_COVER_DAYS') or {}).get('default')) or 21))
            low = mm[pd.to_numeric(mm.get('cover_days'), errors='coerce') < default_cover][['sku','product','cover_days','available']].copy()
            low = low.sort_values('cover_days').head(5)
            alerts = mm[(mm.get('below_reorder') == True)][['sku','product','available']].head(5)
            outputs['inventory_summary'] = {
                'default_cover_days': default_cover,
                'low_cover': low.to_dict(orient='records'),
                'reorder_alerts': alerts.to_dict(orient='records'),
            }
        except Exception as e:
            print(f"[warn] failed to build inventory summary: {e}")

    for a in outputs["actions"]:
        a["evidence"] = evidence_for_action(a, aligned_for_template)
    for p in outputs.get("pilot_actions", []):
        p["evidence"] = evidence_for_action(p, aligned_for_template)

    briefing_out = Path(out_dir) / "briefings" / f"{brand}_briefing.html"
    render_briefing(
        str(Path(Path(__file__).resolve().parent.parent) / "templates"),
        str(briefing_out),
        brand,
        aligned_for_template,
        outputs,
    )

    # --- console summary
    print("Do next:")
    if outputs["actions"]:
        for i, a in enumerate(outputs["actions"], start=1):
            how = "; ".join((a.get("how_to_launch") or [])[:3])
            print(f"{i}) {a.get('title','')} — {a.get('do_this','')}. Steps: {how}. Assets: {a.get('attachment','')}")
    else:
        for p in outputs.get("pilot_actions", []):
            how = "; ".join((p.get("how_to_launch") or [])[:3])
            frac = int((p.get("pilot_audience_fraction", 0.2) or 0.2) * 100)
            budg = int(p.get("pilot_budget_cap", 200) or 200)
            print(f"Pilot) {p.get('title','')} — {p.get('do_this','')}. Pilot {frac}%, Budget ${budg}. Steps: {how}. Assets: {p.get('attachment','')}")

    if outputs["watchlist"]:
        print("Watchlist (needs more data or $; failed ≥1 gate):")
        for w in outputs["watchlist"]:
            failed = ", ".join(w.get("failed", [])) if w.get("failed") else (", ".join(w.get("reasons", [])) or "directional")
            print(f"- {w.get('title','')} — {failed}")

    if outputs["backlog"]:
        print("Backlog (passed all gates; deferred):")
        for b in outputs["backlog"]:
            print(f"- {b.get('title','')} — {b.get('reason','ranked below top actions')}")

    print(f"QA report: {qa_path}")
    print(f"Charts (copied under briefings): {[p for p in outputs['charts']]}")
    print(f"Segments bundle: {outputs['segments_bundle']}")
    print(f"Briefing: {briefing_out}")

    # --- Example: post-implementation tracking (manual)
    # Uncomment and edit the following lines when you want to track an implemented action
    #
    # from src.action_engine import track_implementation_status, track_action_results
    #
    # track_implementation_status(
    #     receipts_dir=str(receipts_dir),
    #     action_id="winback_21_45_base_20250102_143022",
    #     implemented=True,
    #     notes="Launched in Klaviyo",
    #     channels=["email", "sms"],
    #     audience_size=234,
    # )
    #
    # track_action_results(
    #     receipts_dir=str(receipts_dir),
    #     action_id="winback_21_45_base_20250102_143022",
    #     revenue=3250.00,
    #     orders=18,
    #     conversion_rate=0.077,
    #     notes="Slightly below expected but good engagement",
    # )


def main():
    ap = argparse.ArgumentParser()
    # Primary report args
    ap.add_argument("--csv", required=False, help="[Deprecated] Combined orders CSV; use --orders instead")
    ap.add_argument("--orders", required=False, help="Orders CSV (order or line-item level)")
    ap.add_argument("--order-items", required=False, help="Optional order items CSV (line-level)")
    ap.add_argument("--brand", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--inventory", required=False, help="Optional Shopify Inventory CSV path")

    # Tracking: implementation
    ap.add_argument("--track-implemented", action="store_true", help="Update implementation status for an action")
    ap.add_argument("--action-id", type=str, help="Action ID to track (from receipts/actions)")
    ap.add_argument("--implemented", type=str, default="true", help="true|false (default true)")
    ap.add_argument("--channels", type=str, default=None, help="Comma-separated channels, e.g. email,sms")
    ap.add_argument("--audience-size", type=int, default=None, help="Audience size actually targeted")
    ap.add_argument("--notes", type=str, default=None, help="Notes about implementation or results")

    # Tracking: results
    ap.add_argument("--track-results", action="store_true", help="Record results for an action")
    ap.add_argument("--revenue", type=float, default=None, help="Actual revenue realized")
    ap.add_argument("--result-orders", type=int, default=None, help="Actual orders")
    ap.add_argument("--conversion-rate", type=float, default=None, help="Actual conversion rate (0-1)")

    args = ap.parse_args()

    # Tracking-only modes: allow running without CSV/brand
    if args.track_implemented or args.track_results:
        if not args.out:
            ap.error("--out is required to locate receipts when tracking")
        receipts_dir = str(Path(args.out) / "receipts")

        if args.track_implemented:
            if not args.action_id:
                ap.error("--action-id is required with --track-implemented")
            implemented_flag = str(args.implemented).strip().lower() in {"1","true","yes","y","on"}
            channels_list = [c.strip() for c in (args.channels or "").split(",") if c.strip()] if args.channels else None
            res = track_implementation_status(
                receipts_dir=receipts_dir,
                action_id=args.action_id,
                implemented=implemented_flag,
                notes=args.notes,
                channels=channels_list,
                audience_size=args.audience_size,
            )
            print("Implementation tracked:", res.get("action_id"), res.get("status"))

        if args.track_results:
            if not args.action_id:
                ap.error("--action-id is required with --track-results")
            if args.revenue is None:
                ap.error("--revenue is required with --track-results")
            res = track_action_results(
                receipts_dir=receipts_dir,
                action_id=args.action_id,
                revenue=float(args.revenue),
                orders=args.result_orders,
                conversion_rate=args.conversion_rate,
                notes=args.notes,
            )
            print("Results tracked:", res.get("action_id"), res.get("status"), "revenue=", res.get("actual",{}).get("revenue"))
        return

    # Determine source precedence
    orders_path = args.orders or (args.csv if args.csv else None)
    if (args.order_items) and (not args.orders and not args.csv):
        ap.error("--order-items requires --orders (orders table)")
    if not (orders_path and args.brand and args.out):
        ap.error("--orders (or --csv), --brand, --out are required for report generation")

    # Deprecation notice
    if args.csv and not args.orders:
        print("[warn] --csv is deprecated. Use --orders for clearer intent.")

    # Load orders/items with flexible detection, then run
    # We keep the current run() signature using --csv parameter; pass through via temp path var
    run(orders_path, args.brand, args.out, inventory_path=args.inventory, order_items_path=args.order_items)


if __name__ == "__main__":
    main()
