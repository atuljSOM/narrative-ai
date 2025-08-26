
from __future__ import annotations
import argparse
from pathlib import Path
from .utils import get_config, safe_make_dirs, write_json
from .load import load_csv
from .features import compute_features, aligned_periods_summary
from .segments import build_segments
from .charts import generate_charts
from .briefing import render_briefing
from .action_engine import select_actions, write_actions_log
from .copykit import render_copy_for_actions

def run(csv_path: str, brand: str, out_dir: str) -> None:
    cfg = get_config()
    safe_make_dirs(out_dir)
    receipts_dir = str(Path(out_dir)/"receipts"); safe_make_dirs(receipts_dir)
    qa_path = str(Path(receipts_dir)/"qa_report.json")

    df, qa = load_csv(csv_path, qa_out_path=qa_path)
    g = compute_features(df)
    aligned = aligned_periods_summary(g, min_window_n=max(cfg["MIN_N_WINBACK"], 300))

    seg_dir = str(Path(out_dir)/"segments"); seg_files = build_segments(g, cfg["GROSS_MARGIN"], seg_dir)
    chart_dir = str(Path(out_dir)/"charts"); chart_paths = generate_charts(aligned, chart_dir)

    plays = str(Path(Path(__file__).resolve().parent.parent)/"templates"/"playbooks.yml")
    actions = select_actions(g, aligned, cfg, plays, receipts_dir)

    # Render copy assets for selected actions and pilots
    assets_dir = str(Path(out_dir)/"briefings"/"assets")
    selected_for_copy = (actions.get("actions", []) + actions.get("pilot_actions", []))
    for a in selected_for_copy:
        a["brand"] = brand
    copy_assets = render_copy_for_actions(str(Path(Path(__file__).resolve().parent.parent)/"templates"), assets_dir, selected_for_copy)

    summary_path = str(Path(receipts_dir)/"run_summary.json")
    write_json(summary_path, {
        "aligned": aligned, "charts": chart_paths, "segments": seg_files,
        "actions": actions["actions"], "watchlist": actions["watchlist"],
        "pilot_actions": actions.get("pilot_actions", []), "backlog": actions["backlog"],
    })
    write_actions_log(receipts_dir, actions["actions"])

    outputs = {
        "charts": chart_paths,
        "segments_bundle": [s for s in seg_files if s.endswith(".zip")][0] if seg_files else "",
        "actions": actions["actions"], "watchlist": actions["watchlist"],
        "pilot_actions": actions.get("pilot_actions", []), "backlog": actions["backlog"],
        "cfg": cfg, "copy_assets": copy_assets,
    }
    briefing_out = str(Path(out_dir)/"briefings"/f"{brand}_briefing.html")
    render_briefing(str(Path(Path(__file__).resolve().parent.parent)/"templates"), briefing_out, brand, aligned, outputs)

    # Console: To-do checklist
    print("Do next:")
    if outputs["actions"]:
        for i,a in enumerate(outputs["actions"], start=1):
            how = "; ".join(a.get("how_to_launch", [])[:3])
            print(f"{i}) {a['title']} — {a.get('do_this','')}. Steps: {how}. Assets: {a.get('attachment','')}")
    else:
        for p in outputs.get("pilot_actions", []):
            how = "; ".join(p.get("how_to_launch", [])[:3])
            frac = int(p.get("pilot_audience_fraction",0.2)*100)
            budg = int(p.get("pilot_budget_cap",200))
            print(f"Pilot) {p['title']} — {p.get('do_this','')}. Pilot {frac}%, Budget ${budg}. Steps: {how}. Assets: {p.get('attachment','')}")

    if outputs["watchlist"]:
        print("Watchlist:")
        for w in outputs["watchlist"]:
            print(f"- {w['title']} — directional; fails {w['failed']}")

    if outputs["backlog"]:
        print("Backlog:")
        for b in outputs["backlog"]:
            print(f"- {b['title']} — {b.get('reason','')}")

    print(f"QA report: {qa_path}")
    print(f"Charts: {chart_dir}")
    print(f"Segments bundle: {outputs['segments_bundle']}")
    print(f"Briefing: {briefing_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--brand", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); run(args.csv, args.brand, args.out)

if __name__ == "__main__":
    main()
