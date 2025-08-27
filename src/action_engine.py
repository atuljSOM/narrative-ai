from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np, pandas as pd, json

# stats & scoring
from .stats import (
    two_proportion_z_test,   # p-value only
    welch_t_test,            # p-value only
    benjamini_hochberg,      # returns q-values list
    required_n_for_proportion,
    needed_n_for_proportion_delta,
)
from .scoring import (
    compute_score,
    significance_to_score,
    effect_to_score,
    audience_to_score,
    confidence_from_ci,
    financial_to_score,
)

# utils
from .utils import write_json


# ---- Section partition helper (single source of truth for sections) ----
def _partition_candidates(final: list[dict], effort_budget: int = 8):
    """
    Watchlist  := any candidate with failed gates (c['failed'] nonempty) or not allowed
    Pool       := candidates that pass all gates (c['failed'] == [])
    Top        := pick <=3 from Pool subject to effort budget and simple category diversity
    Backlog    := the rest of Pool (passed all gates) not selected
    """
    # classify
    watchlist = [c for c in final if c.get("failed")]  # failed at least one gate
    pool = [c for c in final if not c.get("failed")]   # passed all gates

    # greedy selection with diversity + effort budget
    pool_sorted = sorted(pool, key=lambda x: x.get("score", 0), reverse=True)
    top, used_cats, used_effort = [], set(), 0
    for c in pool_sorted:
        if len(top) >= 3:
            break
        # category diversity: avoid 2 of the same category until we have diversity
        if c.get("category") in used_cats and len(used_cats) < 2:
            continue
        if used_effort + int(c.get("effort", 2)) > effort_budget:
            # mark the reason for deprioritization; item still belongs in backlog (passed all gates)
            c["defer_reason"] = "effort budget"
            continue
        top.append(c)
        used_cats.add(c.get("category"))
        used_effort += int(c.get("effort", 2))

    chosen_ids = {(c.get("id"), c.get("play_id")) for c in top}
    backlog = []
    for c in pool_sorted:
        key = (c.get("id"), c.get("play_id"))
        if key in chosen_ids:
            continue
        # everything in backlog has passed all gates; set default reason if none
        if not c.get("defer_reason"):
            c["defer_reason"] = "ranked below top actions"
        backlog.append(c)

    return top, backlog, watchlist


def load_playbooks(path: str) -> List[Dict[str, Any]]:
    import yaml
    return yaml.safe_load(Path(path).read_text()).get("playbooks", [])

def _gate_and_score(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    passed, failed = [], []
    if candidate["n"] >= candidate["min_n"]: passed.append("min_n")
    else: failed.append("min_n")
   # significance gate: (CI excludes 0) OR (p < 0.05) OR (q < FDR_ALPHA)
    q = candidate.get("q", np.nan)
    p = candidate.get("p", np.nan)
    ci_ok = (
        (candidate.get("ci_low") is not None)
        and (candidate.get("ci_high") is not None)
        and (candidate["ci_low"] * candidate["ci_high"] > 0)
    )
    sig_ok = ci_ok or (not np.isnan(p) and p < 0.05) or (not np.isnan(q) and q < cfg["FDR_ALPHA"])

    if sig_ok: passed.append("significance")
    else: failed.append("significance")
    if abs(candidate["effect_abs"]) >= candidate["effect_floor"]: passed.append("effect_floor")
    else: failed.append("effect_floor")
    if candidate["expected_$"] >= cfg["FINANCIAL_FLOOR"]: passed.append("financial_floor")
    else: failed.append("financial_floor")

    significance_score = significance_to_score(candidate.get("p", np.nan), candidate.get("q", np.nan), cfg["FDR_ALPHA"])
    effect_score = effect_to_score(candidate["effect_abs"], candidate["effect_floor"])
    audience_score = audience_to_score(candidate["n"], candidate["min_n"])
    confidence_score = confidence_from_ci(candidate.get("ci_low", np.nan), candidate.get("ci_high", np.nan))
    financial_score = financial_to_score(candidate["expected_$"], cfg["FINANCIAL_FLOOR"])
    total = compute_score(financial_score, significance_score, effect_score, confidence_score, audience_score)

    cand = candidate.copy()
    cand["passed"], cand["failed"], cand["score"] = passed, failed, total
    cand["scores_breakdown"] = {
        "financial": financial_score, "significance": significance_score, "effect_size": effect_score,
        "confidence": confidence_score, "audience_size": audience_score,
    }
        # --- Tiering: Actions / Watchlist / No call ---
    # Current: Actions = all gates pass
    # New: Watchlist = directional signal but fails at least one gate
    #  - any of these qualifies as "directional":
    #    (a) n >= 0.5*min_n, OR
    #    (b) p < 0.25 (raw) OR
    #    (c) abs(effect) >= 0.5*effect_floor OR
    #    (d) expected_$ >= 0.5*FINANCIAL_FLOOR
    directional = (
        (cand["n"] >= 0.5 * candidate["min_n"]) or
        ((candidate.get("p") is not None) and (not np.isnan(candidate.get("p"))) and (candidate["p"] < 0.25)) or
        (abs(candidate["effect_abs"]) >= 0.5 * candidate["effect_floor"]) or
        (candidate["expected_$"] >= 0.5 * cfg["FINANCIAL_FLOOR"])
    )

    if len(failed) == 0:
        tier = "Actions"
    elif directional:
        tier = "Watchlist"
    else:
        tier = "No call"

    cand["tier"] = tier

    # --- Human-readable reasons for Watchlist/Backlog cards ---
    reasons = []
    if "min_n" in failed:
        need = max(0, int(candidate["min_n"] - candidate["n"]))
        reasons.append(f"needs ~{need} more orders (min_n={candidate['min_n']})")
    if "significance" in failed:
        p = candidate.get("p", np.nan)
        q = candidate.get("q", np.nan)
        reasons.append(f"not yet significant (p={p:.3f}, q={q:.3f})")
    if "effect_floor" in failed:
        reasons.append(f"effect below floor (Δ={candidate['effect_abs']:+.3%} vs floor {candidate['effect_floor']:.1%})")
    if "financial_floor" in failed:
        short = max(0.0, float(cfg['FINANCIAL_FLOOR'] - candidate['expected_$']))
        reasons.append(f"fails financial floor by ${short:,.0f} (needs ≥ ${cfg['FINANCIAL_FLOOR']:,.0f})")

    cand["reasons"] = reasons

    return cand


def _compute_candidates(g: pd.DataFrame, aligned: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    gross_margin = cfg["GROSS_MARGIN"]

    # Repeat rate
   # Repeat rate (recent vs prior window)
    x1 = int(round((aligned["recent_repeat_rate"] or 0) * (aligned["recent_n"] or 0)))
    x2 = int(round((aligned["prior_repeat_rate"]  or 0) * (aligned["prior_n"]  or 0)))
    n1, n2 = int(aligned["recent_n"] or 0), int(aligned["prior_n"] or 0)

    pval = two_proportion_z_test(x1, n1, x2, n2)
    rate_recent = (x1 / n1) if n1 else 0.0
    rate_prior  = (x2 / n2) if n2 else 0.0
    effect_pts  = rate_recent - rate_prior  # absolute delta in points (e.g., +0.024)

    expected = max(0.0, effect_pts) * (n1) * (aligned["prior_repeat_rate"] or 0.15) * gross_margin

    cands.append({
        "id": "repeat_rate_improve",
        "play_id": "winback_21_45",
        "metric": "repeat_rate",
        "n": n1 + n2,
        "effect_abs": effect_pts,
        "p": pval,
        "q": np.nan,                     # set later by BH
        "ci_low": None, "ci_high": None, # (optional) add CI later if you implement it
        "expected_$": expected,
        "min_n": cfg["MIN_N_WINBACK"],
        "effect_floor": cfg["REPEAT_PTS_FLOOR"],
        "rationale": f"Repeat share {rate_recent:.1%} vs {rate_prior:.1%} (Δ {effect_pts:+.1%}).",
        "audience_size": n1,
        "attachment": "segment_winback_21_45.csv",
        "baseline_rate": rate_prior or 0.15,
    })


    # AOV
    # AOV (Welch t-test)
    maxd = pd.to_datetime(g["Created at"]).max()
    start = maxd - pd.Timedelta(days=aligned["window_days"] - 1)
    rec = g[g["Created at"] >= start]["AOV"].astype(float).values
    pri = g[(g["Created at"] < start) & (g["Created at"] >= start - pd.Timedelta(days=aligned["window_days"]))]["AOV"].astype(float).values

    if rec.size > 0 and pri.size > 0:
        pval = welch_t_test(rec, pri)
        m1, m2 = float(np.mean(rec)), float(np.mean(pri))
        effect_pct = (m1 - m2) / m2 if m2 else 0.0
        expected = max(0.0, effect_pct) * (aligned["recent_n"] or 0) * (aligned["prior_repeat_rate"] or 0.15) * (aligned["prior_aov"] or m2 or 0.0) * gross_margin

        cands.append({
            "id": "aov_increase",
            "play_id": "bestseller_amplify",
            "metric": "aov",
            "n": int(rec.size + pri.size),
            "effect_abs": effect_pct,
            "p": pval,
            "q": np.nan,
            "ci_low": None, "ci_high": None,
            "expected_$": expected,
            "min_n": cfg["MIN_N_SKU"],
            "effect_floor": cfg["AOV_EFFECT_FLOOR"],
            "rationale": f"AOV {m1:.2f} vs {m2:.2f} (Δ {effect_pct:+.1%}).",
            "audience_size": int(rec.size),
            "attachment": "segment_bestseller_amplify.csv",
            "baseline_rate": aligned["prior_repeat_rate"] or 0.15,
        })


    # Discount rate share (>=5% discounted orders)
    # Discount rate share (>=5% discounted orders)
    rec_disc = g[g["Created at"] >= start]["discount_rate"].fillna(0).astype(float).values
    pri_disc = g[(g["Created at"] < start) & (g["Created at"] >= start - pd.Timedelta(days=aligned["window_days"]))]["discount_rate"].fillna(0).astype(float).values

    x1, n1 = int(np.sum(rec_disc >= 0.05)), int(rec_disc.size)
    x2, n2 = int(np.sum(pri_disc >= 0.05)), int(pri_disc.size)
    if n1 > 0 and n2 > 0:
        pval2 = two_proportion_z_test(x1, n1, x2, n2)
        # effect is negative if discount share increased (we want *reduction*)
        effect_pts2 = (x2 / n2) - (x1 / n1)  # reduction is positive if recent < prior
        expected2 = max(0.0, effect_pts2) * n1 * 0.5 * gross_margin * (float(np.nanmean(g["AOV"])) if not np.isnan(np.nanmean(g["AOV"])) else 0.0)

        cands.append({
            "id": "discount_hygiene",
            "play_id": "discount_hygiene",
            "metric": "discount_rate",
            "n": n1 + n2,
            "effect_abs": effect_pts2,
            "p": pval2,
            "q": np.nan,
            "ci_low": None, "ci_high": None,
            "expected_$": expected2,
            "min_n": cfg["MIN_N_SKU"],
            "effect_floor": cfg["DISCOUNT_PTS_FLOOR"],
            "rationale": f"Discount share {x1/(n1 or 1):.1%} vs {x2/(n2 or 1):.1%} (Δ {effect_pts2:+.1%} reduction).",
            "audience_size": n1,
            "attachment": "segment_discount_hygiene.csv",
            "baseline_rate": aligned["prior_repeat_rate"] or 0.15,
        })

    return cands

def select_actions(g: pd.DataFrame, aligned: Dict[str, Any], cfg: Dict[str, Any], playbooks_path: str, receipts_dir: str) -> Dict[str, Any]:
    plays = {p["id"]: p for p in load_playbooks(playbooks_path)}
    cands = _compute_candidates(g, aligned, cfg)

    # FDR adjust (q-values)
    # FDR adjust (q-values)
    p_list = []
    for c in cands:
        p = c.get("p", np.nan)
        p_list.append(1.0 if np.isnan(p) else float(p))

    if any(not np.isnan(c.get("p", np.nan)) for c in cands):
        qvals = benjamini_hochberg(p_list)  # returns list aligned to p_list order
        for i, c in enumerate(cands):
            c["q"] = qvals[i]


    # Gate + score + meta attach
    final = []
    for c in cands:
        meta = plays.get(c["play_id"], {})
        c["category"] = meta.get("category", "general")
        c["title"] = meta.get("title", c["play_id"])
        for k in ["do_this","targeting","channels","cadence","offer","copy_snippets","assets","how_to_launch",
                  "success_criteria","risks_mitigations","owner_suggested","time_to_set_up_minutes","holdout_plan"]:
            c[k] = meta.get(k)
        c["effort"] = meta.get("effort", 2)
        c["risk"] = meta.get("risk", 2)

        cand = _gate_and_score(c, cfg)

        # Confidence label + expected range
        if len(cand["failed"]) == 0:
            cand["confidence_label"] = "High"
        elif ("min_n" in cand["passed"]) and ("significance" in cand["failed"]) and ("effect_floor" not in cand["failed"]) and ("financial_floor" not in cand["failed"]):
            cand["confidence_label"] = "Medium"
        else:
            cand["confidence_label"] = "Low"
        expv = cand.get("expected_$", 0) or 0
        cand["expected_range"] = [round(expv * 0.6, 2), round(expv * 1.3, 2)]

        final.append(cand)

    # Partition
    # Partition (single source of truth)
    budget = cfg.get("EFFORT_BUDGET", 8)
    top_actions, backlog, watchlist = _partition_candidates(final, effort_budget=budget)

    out = {
        "actions": top_actions,     # passed all gates + selected under budget/diversity
        "watchlist": watchlist,     # failed at least one gate (no attachments)
        "no_call": [],              # keep for compatibility if you use it elsewhere
        "backlog": [],              # passed all gates but deferred
        "pilot_actions": []
    }

    # Backlog: passed all gates, deferred (set a reason if not already present)
    for c in backlog:
        out["backlog"].append({**c, "reason": c.get("defer_reason", "ranked below top actions")})

    # Pilot fallback (only if nothing made it into Top Actions)
    if len(out["actions"]) == 0 and len(final) > 0:
        # choose the highest-score candidate as a pilot, even if it failed a gate (clearly labeled)
        pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
        n_needed = pilot.get("min_n", 0)
        if pilot.get("metric") in ("repeat_rate", "discount_rate"):
            p = pilot.get("baseline_rate", 0.15) or 0.15
            delta = pilot.get("effect_floor", 0.02) or 0.02
            # required_n_for_proportion should already exist in your stats/helpers
            n_needed = required_n_for_proportion(p, delta, alpha=0.05, power=0.8)

        out["pilot_actions"] = [{
            **pilot,
            "tier": "Pilot",
            "pilot_audience_fraction": cfg.get("PILOT_AUDIENCE_FRACTION", 0.2),
            "pilot_budget_cap": cfg.get("PILOT_BUDGET_CAP", 200.0),
            "n_needed": int(n_needed),
            "decision_rule": "Graduate to full rollout if CI excludes 0 or q ≤ α at 14 days; else rollback.",
            "confidence_label": "Low",
            "expected_range": [
                round((pilot.get("expected_$", 0) or 0) * 0.6, 2),
                round((pilot.get("expected_$", 0) or 0) * 1.3, 2),
            ],
        }]

    return out

def write_actions_log(receipts_dir: str, actions: List[Dict[str, Any]]) -> None:
    log_path = Path(receipts_dir)/"actions_log.json"
    log = []
    if log_path.exists():
        log = json.loads(log_path.read_text())
    for a in actions:
        log.append({"play_id": a["play_id"], "title": a["title"]})
    write_json(str(log_path), log)

def tipover_for_financial(shortfall: float, segment_size: int, baseline_rate: float, gross_margin: float) -> dict:
    """
    Given a $ shortfall to the financial floor, estimate how many additional reachable
    customers you need to clear the floor, using baseline conversion and GM.
    Returns a dict with the missing dollars and an order-of-magnitude customer count.
    """
    br = max(float(baseline_rate), 1e-9)
    gm = max(float(gross_margin),  1e-9)
    add_customers = int(np.ceil(float(shortfall) / (br * gm)))
    return {"needed_$": round(float(shortfall), 2), "add_customers≈": max(add_customers, 0)}

def tipover_for_significance(p_base: float, current_n: int, alpha: float, power: float, target_delta: float) -> dict:
    """
    For a two-proportion test, compute additional per-group N needed to detect target_delta (absolute)
    at the given alpha/power. Returns 0 if current_n already meets/exceeds the requirement.
    """
    need_n = int(needed_n_for_proportion_delta(float(p_base), float(target_delta), float(alpha), float(power)))
    return {"needed_n_per_group": max(0, need_n - int(current_n))}
