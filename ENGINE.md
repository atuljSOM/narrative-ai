# Aura PMUX Build — Action Engine (Monthly)

## Purpose
- Recommend a realistic monthly action plan to drive incremental revenue for beauty and supplement brands.
- Balance rigor (stats, power, floors) with practicality (fatigue, cooldowns, interactions, effort, audience overlap).
- Produce explainable outputs (briefing HTML, segments, charts) and track outcomes for accuracy calibration.

## Data & Features
- Inputs (order-level): `Created at`, `Name` (order id), `customer_id`, `Subtotal`, `Total Discount`, `Shipping`, `Taxes`, `net_sales`, `AOV`, `discount_rate`, `is_repeat`, `units_per_order`, `lineitem_any`, `days_since_last`.
- Canonical revenue: `net_sales` is computed per-order as `Subtotal - Total Discount`; if `Subtotal`/`Total Discount` are missing per row, we fallback per-order to `Total - Shipping - Taxes`. If orders-level fields are missing entirely, we fallback to line-items aggregation restricted to the same orders/time window. We always prefer the same method and record which method was used.
- Feature pipelines:
  - KPI windows and deltas with optional seasonal adjustment (expectations stored in metadata; actuals never overridden).
  - Segment audiences per play (winback/dormant/sample/subscription/etc.).
  - LTV signal for light preference on no-discount variants.

Key code:
- KPI snapshot and window selection: `src/utils.py`
- Features and aligned periods: `src/features.py`
- Engine and selection: `src/action_engine.py`
- Charts: `src/charts.py`
- Briefing rendering: `src/briefing.py`, `templates/briefing.html.j2`

## Metrics Reference (v2)
- net_sales: Sum of per-order `"_order_net"` where `"_order_net" = Subtotal - Total Discount` (fallback per order to `Total - Shipping - Taxes`). Line-items aggregate used only when order-level fields are absent. Metadata records method and consistency vs alternatives.
- orders: Unique orders in window (by `Name`).
- aov: `net_sales / orders` (computed from canonical values).
- discount_rate: Sum(`Total Discount`) ÷ Sum(`Subtotal`) for the window when available.
- repeat_rate_within_window: Share of identified customers in the window who placed 2+ orders inside the same window. Purpose: short-cycle engagement; used for performance comparisons and action triggers.
- returning_customer_share: Share of identified customers in the window who have any order before the window start. Purpose: acquisition/mix understanding.
- new_customer_rate: `1 − returning_customer_share`. Purpose: growth tracking.

Back-compat aliases:
- `repeat_share` → `repeat_rate_within_window`
- `repeat_rate` → `repeat_rate_within_window`
- `returning_rate` → `returning_customer_share`

Snapshot metadata:
- `meta.metric_version = "v2_repeat_metrics"`
- Seasonal expectations under `seasonal_expected` (orders/net sales recent/prior, expected lifts, surprise vs expected).
- Netsales transparency: `recent_netsales_method`, `prior_netsales_method`, `*_netsales_alt_diffs` (pct diffs vs `subtotal_minus_discount`, `total_minus_shipping_taxes`, `line_items_aggregate`), and `*_netsales_consistency_flag` if any alternative differs by >10%.

## Monthly Cadence & Scaling
- Measurement window: 28 days by default for tracking and pilot evaluation.
- Expected impact is expressed in monthly units.
- Scaling from per-week proxy to monthly uses:
  - Fatigue schedule [1.00, 0.85, 0.70, 0.60] → 3.15× instead of 4×.
  - Seasonality nudge: ratio of L7 weekly vs L28 average weekly, clamped to [0.9, 1.1].
  - Per-action saturation cap: ≤ 35% of L28 net sales.

Rationale: later-week attention/campaign fatigue, light month-within variability, and bound against implausibly large single-action claims.

## Candidate Actions (Math & Stats)
Each candidate produces: metric, n, effect_abs, p-value, CI (when possible), expected_$, floors, and attachments.

1) Repeat (Within Window) Improve → `winback_21_45`
- Construct x1/n1 (recent L28) and x2/n2 (prior L28) where x = count of customers with ≥2 orders inside each window; n = identified customers in-window.
- Hypothesis: Δ = p1 − p2 > 0
  - Test: two-proportion z-test p-value (Fisher fallback in extremes)
  - CI: Wilson CI per period; conservative ΔCI = [lo1 − hi0, hi1 − lo0]
- Expected_$ ≈ max(0, Δ) × recent_n × prior_repeat_rate × gross_margin

2) AOV Increase → `bestseller_amplify`
- Compare AOV_recent vs AOV_prior via Welch t-test (two-sided p).
- Effect_abs = (AOV_recent − AOV_prior)/AOV_prior.
- Expected_$ ≈ max(0, effect%) × recent_n × prior_repeat_rate × prior_aov × gross_margin

3) Discount Hygiene → `discount_hygiene`
- x1/n1 = share of orders with discount_rate ≥ 5% in recent; x2/n2 prior.
- We want reduction: effect_abs = (x2/n2) − (x1/n1) (positive is good).
- Test: two-proportion p-value; CI via Wilson ΔCI.
- Expected_$: conservative guardrail using AOV × GM and weekly baseline caps.

4) Subscription Nudge → `subscription_nudge`
- Audience: customers with ≥ threshold product orders in 90d (threshold by vertical/product detection).
- Expected_$ (weekly proxy): audience weekly orders × AOV × GM × uplift (≈0.25 over ~12 weeks → ~0.021/week) with weekly caps, then monthly scaling.
- Power gate: min_n = max(MIN_N_SKU, required_n_for_proportion(baseline_conv, Δ=0.05, α=0.05, power=0.8)).
  - baseline_conv approximated from two historic windows when available; default 0.15.

5) Sample → Full-size → `sample_to_full`
- Audience: sample/travel buyers in 14–21d without full-size later.
- Expected_$: weekly proxy ≈ 35% over ~3 weeks; capped; then monthly scaling.
- Empirical p-value and Wilson ΔCI from earlier comparison windows (if possible).

6) “Empty Bottle” Reorder → `empty_bottle`
- Audience: perfume-like size parsing (30/50/100ml) near depletion window via `days_since_last` vs typical supply.
- Expected_$: weekly reminders × ~10% conv × AOV × GM; capped; then monthly scaling.
- Empirical p-value and Wilson ΔCI from prior near-depletion cohorts (if possible).

Notes
- Additional plays can be added similarly; ensure you set floors, power if proportions, and weekly → monthly scaling.

## Gating & Scoring
- Gates (must pass all for “Actions” tier):
  - min_n: audience size threshold (env-driven and/or power-based)
  - significance: (CI excludes 0) OR (p < 0.05) OR (q < FDR_ALPHA) after BH correction across base candidates
  - effect_floor: absolute effect above floor
  - financial_floor: expected_$ ≥ FINANCIAL_FLOOR (auto or fixed)
- Directional “Watchlist”: passes any soft criterion (e.g., ≥ 0.5× min_n, p < 0.25, ≥ 0.5× effect_floor, or ≥ 0.5× financial floor).
- Scoring: weighted blend of financial, significance, effect size, CI tightness, and audience size.

## Variants, Policy & LTV Preferences
- Variants from `templates/playbooks.yml`: define `offer_type`, `lift_multiplier`, optional cost (`percent_of_aov` or `flat_per_order`).
- Expected_$ per variant: base × lift − incentive_cost.
- LTV preference: small nudge toward no-discount variant for top-decile LTV audiences.
- Policy gating: optional rules (e.g., discount caps, channel caps) to disable ineligible variants.

## Cooldowns & Novelty
- Week-aware cooldown by play family; skip if within `cooldown_weeks`.
- Novelty penalty for 0–2 weeks since last use of same variant.

## Selection & Diversity
- Partition pool → Top (≤3), Backlog, Watchlist.
- Enforce one variant per play, soft category diversity (≥2 categories if possible), and total effort ≤ `EFFORT_BUDGET`.
- Pilot fallback: if none pass, highest-score candidate proposed as pilot (28d decision rule).

## Overlap & Interaction Safeguards
- Audience overlap (dedupe impacts):
  - Read `segments/segment_*.csv` for selected actions (using their `attachment`).
  - Sequentially reduce later actions’ expected_$ by overlap ratio with prior combined audience.
  - Annotate overlap% and effective audience on cards.
- Campaign interactions (configurable):
  - Pairwise dampening factors from `.env` (`INTERACTION_FACTORS`), e.g. `discount_hygiene->winback_21_45:0.9`.
  - Multiply later actions’ expected_$ by product of applicable factors; annotate on cards.

## Forecast Chart (Monthly Units)
- Baseline: labeled as `4× L7`, `L28`, or `L56 ÷ 2` source.
- Combined lift: adjusts for channel overlap and position-based diminishing returns [1.0, 0.9, 0.8].
- Portfolio cap: total monthly lift ≤ 50% of baseline.
- Clarifying subtitle: units, baseline source, and interaction assumptions.

## Action Tracking & Accuracy
- Each recommended action is registered with an ID and predicted values; upon implementation and completion (28 days), actuals recorded.
- Accuracy metrics (e.g., revenue accuracy %) and performance summaries (aggregate/median) are computed.
- Pending items and overdue detection help operational follow-up.

## Validation Enhancements
- Inventory validation:
  - Schema presence, snapshot freshness (INVENTORY_MAX_AGE_DAYS), count of low-cover SKUs.
  - Reorder point alerts (below reorder with no incoming) highlighted.
  - Seasonal stockout risk (heuristic): in peak months (Nov/Dec), SKUs with cover < 14d and daily_velocity ≥ 1 are flagged.

- Metric consistency:
  - `new_customer_rate` ≈ `1 − returning_customer_share` (tolerance ~1–2pp, configurable).
  - `orders × aov` close to `net_sales` within 10% tolerance.
  - Netsales method consistency flags surfaced in debug snapshot for investigation.

## Configuration Reference (.env)
- Thresholds: `MIN_N_WINBACK`, `MIN_N_SKU`, `AOV_EFFECT_FLOOR`, `REPEAT_PTS_FLOOR`, `DISCOUNT_PTS_FLOOR`.
- Financials: `FINANCIAL_FLOOR_MODE` (auto|fixed), `FINANCIAL_FLOOR_FIXED`, `FINANCIAL_FLOOR`, `GROSS_MARGIN`.
- Windows & display: `WINDOW_POLICY` (auto|l7|l28|l56), `L7_MIN_ORDERS`, `L28_MIN_ORDERS`, `SHOW_L7`.
- Pilot: `PILOT_AUDIENCE_FRACTION`, `PILOT_BUDGET_CAP`.
- Seasonality: `SEASONAL_ADJUST`, `SEASONAL_PERIOD`.
- Vertical: `VERTICAL_MODE` (beauty|supplements|mixed).
- Interactions: `INTERACTION_FACTORS` (JSON or CSV for pairwise dampening), e.g.
  - `{"discount_hygiene->winback_21_45":0.9, "winback_21_45->dormant_multibuyers_60_120":0.92}`

## Edge Cases & Guards
- Small n: Fisher’s exact fallback; Wilson CI spans [0,1] if n=0; directional watchlist if promising but underpowered.
- Missing data: template elides stats when missing; charts and KPI sections degrade gracefully.
- Caps & floors: weekly caps pre-scaling; per-action monthly cap 35% baseline; chart portfolio cap 50% baseline; financial floor gating.
- Cooldowns & novelty penalties avoid repetition and encourage variety.

## Reasoning Summary
- Statistical rigor: p-values, FDR, Wilson CIs, and power checks reduce false positives.
- Financial discipline: floors and gross margin scaling ensure monetary significance.
- Operational realism: fatigue, seasonality, saturation, audience overlap, and campaign interactions avoid inflated forecasts.
- Explainability: every selected action shows numeric evidence, steps, segments, and confidence.

## Examples
- Repeat (within window) rise: Δ=+2.5 pts; Wilson ΔCI includes 0 → may be Watchlist unless financials and min_n are strong.
- Subscription nudge: audience 180 with baseline 15%; power requires ≈246 to detect +5 pts → min_n fails; rationale lists n_needed.
- Combined plan: winback + discount_hygiene → winback expected_$ dampened by configured interaction factor (e.g., 0.90) and overlap deduction.

## Where to Look in Code
- Engine & selection logic: `narrative_ai_mvp_stats_pmux_build/src/action_engine.py`
- Config & parsers: `narrative_ai_mvp_stats_pmux_build/src/utils.py`
- Stats helpers: `narrative_ai_mvp_stats_pmux_build/src/stats.py`
- Charts: `narrative_ai_mvp_stats_pmux_build/src/charts.py`
- Briefing template: `narrative_ai_mvp_stats_pmux_build/templates/briefing.html.j2`
- Playbooks metadata: `narrative_ai_mvp_stats_pmux_build/templates/playbooks.yml`

---
Suggestions welcome: if a brand favors SMS-heavy engagement or has strict discount policies, set `INTERACTION_FACTORS` and policy caps accordingly, and consider tuning the fatigue schedule or portfolio cap.
