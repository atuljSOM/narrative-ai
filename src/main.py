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
)
from .load import load_csv
from .features import compute_features, aligned_periods_summary
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


def run(csv_path: str, brand: str, out_dir: str) -> None:
    cfg = get_config()

    # --- output dirs
    safe_make_dirs(out_dir)
    receipts_dir = Path(out_dir) / "receipts"
    safe_make_dirs(str(receipts_dir))
    qa_path = str(receipts_dir / "qa_report.json")

    # --- load & features
    df, qa = load_csv(csv_path, qa_out_path=qa_path)
    g = compute_features(df)

    # summary used by charts
    aligned = aligned_periods_summary(g, min_window_n=max(cfg.get("MIN_N_WINBACK", 150), 300))

    # segments
    seg_dir = Path(out_dir) / "segments"
    seg_files = build_segments(g, cfg.get("GROSS_MARGIN", 0.70), str(seg_dir), cfg)

    # (charts generation moved to after actions are selected)

    # --- KPI snapshot with deltas (use RAW df)
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

    # --- actions
    plays = str(Path(Path(__file__).resolve().parent.parent) / "templates" / "playbooks.yml")
    actions = select_actions(g, aligned, cfg, plays, str(receipts_dir))

    receipts = build_receipts(aligned_for_template, actions)

    # --- Charts (new): generate with feature df and copy near the HTML ---
    chart_out_dir = Path(out_dir) / "charts"
    chart_data = generate_charts(
        g=g,
        aligned=aligned_for_template,
        actions=actions,
        out_dir=str(chart_out_dir),
        df=df,
        chosen_window=str(cfg.get("CHOSEN_WINDOW", "L28")),
        charts_mode=str(cfg.get("CHARTS_MODE", "detailed"))
    ) or {}

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
        qa=qa
    )
    
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
    write_json(str(summary_path), {
        "aligned": aligned,
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
        "cfg": cfg,
        "copy_assets": copy_assets,
        "receipts": receipts,
        "validation_html": validation_html,  # Pass to template
        "validation_results": validation_results,  # Pass full results too
        "performance_summary": performance_summary,
        "pending_actions": pending_actions,
    }

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
    ap.add_argument("--csv", required=False)
    ap.add_argument("--brand", required=False)
    ap.add_argument("--out", required=False)

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
    ap.add_argument("--orders", type=int, default=None, help="Actual orders")
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
                orders=args.orders,
                conversion_rate=args.conversion_rate,
                notes=args.notes,
            )
            print("Results tracked:", res.get("action_id"), res.get("status"), "revenue=", res.get("actual",{}).get("revenue"))
        return

    # Regular report run requires csv/brand/out
    if not (args.csv and args.brand and args.out):
        ap.error("--csv, --brand, --out are required for report generation")
    run(args.csv, args.brand, args.out)


if __name__ == "__main__":
    main()
