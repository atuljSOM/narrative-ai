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
from .action_engine import select_actions, write_actions_log, build_receipts, evidence_for_action
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
    seg_files = build_segments(g, cfg.get("GROSS_MARGIN", 0.70), str(seg_dir))

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
        chosen_window=str(cfg.get("CHOSEN_WINDOW", "L28"))
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
        "validation_results": validation_results  # Pass full results too
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--brand", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.csv, args.brand, args.out)


if __name__ == "__main__":
    main()
