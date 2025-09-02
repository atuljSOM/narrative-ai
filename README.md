
# Narrative.AI — Concierge MVP — PM Action-First UX

- Plain-English action cards with steps, targeting, channels, offers, assets.
- Evidence is behind a collapsible 'View evidence' block.
- Console prints a to-do checklist.

## Run
```bash
python -m src.main --csv tests/data/orders_sample.csv --brand ACME --out analysis/ACME
```

## Small/Micro presets
Add a `.env` with relaxed thresholds (see earlier guidance) to ensure at least a Pilot shows for tiny datasets.

## Product Performance chart too crowded?
- Use the compact mode to simplify the "Product Performance" chart to the two most actionable visuals (Top Velocity and Subscription Readiness):

  - Set in your environment or `.env`:
    - `CHARTS_MODE=compact`

  - Default is `detailed` (a 2x2 grid). In `compact`/`minimal` mode the chart becomes a cleaner 1x2 layout and limits to top 5 products.

## Beauty vs Supplements verticals
- You can guide vertical behavior (winback/dormant windows, subscription thresholds) via env:
  - `VERTICAL_MODE=beauty | supplements | mixed` (default: `mixed` with product auto-detection)

## KPI windows
- To reduce noise for Beauty/Supplements, you can hide the L7 card and focus on L28 vs prior L28:
  - Set `SHOW_L7=False` in `.env`
