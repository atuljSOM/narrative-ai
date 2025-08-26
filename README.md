
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
