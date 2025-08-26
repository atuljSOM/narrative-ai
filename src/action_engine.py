
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np, pandas as pd, json
from .stats import two_proportion_test, welch_t_test, benjamini_hochberg, wilson_ci, required_n_for_proportion
from .scoring import compute_score, significance_to_score, effect_to_score, audience_to_score, confidence_from_ci, financial_to_score
from .utils import read_json, write_json

def load_playbooks(path: str) -> List[Dict[str, Any]]:
    import yaml
    return yaml.safe_load(Path(path).read_text()).get("playbooks", [])

def _gate_and_score(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    passed, failed = [], []
    if candidate["n"] >= candidate["min_n"]: passed.append("min_n")
    else: failed.append("min_n")
    sig_ok = ((not np.isnan(candidate.get("q", np.nan))) and (candidate["q"] < 0.05)) or (
        (candidate.get("ci_low") is not None and candidate.get("ci_high") is not None) and (candidate["ci_low"] * candidate["ci_high"] > 0)
    )
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
    x1 = int(round((aligned["recent_repeat_rate"] or 0)*aligned["recent_n"])); x2 = int(round((aligned["prior_repeat_rate"] or 0)*aligned["prior_n"]))
    pr = two_proportion_test(x1, aligned["recent_n"] or 1, x2, aligned["prior_n"] or 1)
    effect_pts = pr.diff
    expected = max(0.0, effect_pts) * aligned["recent_n"] * (aligned["prior_repeat_rate"] or 0.15) * gross_margin
    cands.append({
        "id":"repeat_rate_improve","play_id":"winback_21_45","metric":"repeat_rate",
        "n": aligned["recent_n"], "effect_abs": effect_pts, "p": pr.p_value, "q": np.nan,
        "ci_low": pr.ci_low, "ci_high": pr.ci_high, "expected_$": expected,
        "min_n": cfg["MIN_N_WINBACK"], "effect_floor": cfg["REPEAT_PTS_FLOOR"],
        "rationale": f"Repeat share {aligned['recent_repeat_rate']:.1%} vs {aligned['prior_repeat_rate']:.1%} (Δ {effect_pts:+.1%}).",
        "audience_size": aligned["recent_n"], "attachment":"segment_winback_21_45.csv",
        "baseline_rate": aligned["prior_repeat_rate"] or 0.15,
    })

    # AOV
    maxd = pd.to_datetime(g["Created at"]).max()
    start = maxd - pd.Timedelta(days=aligned["window_days"]-1)
    rec = g[g["Created at"]>=start]["AOV"].values
    pri = g[(g["Created at"]<start)&(g["Created at"]>= start - pd.Timedelta(days=aligned["window_days"]))]["AOV"].values
    if len(rec)>0 and len(pri)>0:
        mt = welch_t_test(rec, pri)
        aov_prior = aligned["prior_aov"] or 0.0
        effect_pct = (mt.m1 - mt.m2)/mt.m2 if mt.m2 else 0.0
        expected = max(0.0, effect_pct) * aligned["recent_n"] * (aligned["prior_repeat_rate"] or 0.15) * (aov_prior) * gross_margin
        cands.append({
            "id":"aov_increase","play_id":"bestseller_amplify","metric":"aov",
            "n": len(rec)+len(pri), "effect_abs": effect_pct, "p": mt.p_value, "q": np.nan,
            "ci_low": mt.ci_low, "ci_high": mt.ci_high, "expected_$": expected,
            "min_n": cfg["MIN_N_SKU"], "effect_floor": cfg["AOV_EFFECT_FLOOR"],
            "rationale": f"AOV {aligned['recent_aov']:.2f} vs {aligned['prior_aov']:.2f} (Δ {effect_pct:+.1%}).",
            "audience_size": len(rec), "attachment":"segment_bestseller_amplify.csv",
            "baseline_rate": aligned["prior_repeat_rate"] or 0.15,
        })

    # Discount rate share (>=5% discounted orders)
    start = maxd - pd.Timedelta(days=aligned["window_days"]-1)
    rec_disc = g[g["Created at"]>=start]["discount_rate"].fillna(0).values
    pri_disc = g[(g["Created at"]<start)&(g["Created at"]>= start - pd.Timedelta(days=aligned["window_days"]))]["discount_rate"].fillna(0).values
    x1 = int(np.sum(rec_disc>=0.05)); n1 = len(rec_disc); x2 = int(np.sum(pri_disc>=0.05)); n2 = len(pri_disc)
    if n1>0 and n2>0:
        pr2 = two_proportion_test(x1, n1, x2, n2)
        effect_pts2 = (x1/n1) - (x2/n2)
        expected2 = max(0.0, -effect_pts2) * n1 * 0.5 * gross_margin * (np.nanmean(g["AOV"]) or 0)
        cands.append({
            "id":"discount_hygiene","play_id":"discount_hygiene","metric":"discount_rate",
            "n": n1+n2, "effect_abs": -effect_pts2, "p": pr2.p_value, "q": np.nan,
            "ci_low": pr2.ci_low, "ci_high": pr2.ci_high, "expected_$": expected2,
            "min_n": cfg["MIN_N_SKU"], "effect_floor": cfg["DISCOUNT_PTS_FLOOR"],
            "rationale": f"Discount share {x1/(n1 or 1):.1%} vs {x2/(n2 or 1):.1%} (Δ {-effect_pts2:+.1%} reduction).",
            "audience_size": n1, "attachment":"segment_discount_hygiene.csv",
            "baseline_rate": aligned["prior_repeat_rate"] or 0.15,
        })
    return cands

def select_actions(g: pd.DataFrame, aligned: Dict[str, Any], cfg: Dict[str, Any], playbooks_path: str, receipts_dir: str) -> Dict[str, Any]:
    plays = {p["id"]: p for p in load_playbooks(playbooks_path)}
    cands = _compute_candidates(g, aligned, cfg)

    # FDR adjust (q-values)
    pvals = [c.get("p", np.nan) for c in cands]
    if sum([0 if np.isnan(p) else 1 for p in pvals]) > 0:
        qvals, _ = benjamini_hochberg([p if not np.isnan(p) else 1.0 for p in pvals], alpha=cfg["FDR_ALPHA"])
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
    out = {"actions": [], "watchlist": [], "no_call": [], "backlog": [], "pilot_actions": []}
    for c in final:
        if c["tier"] == "Actions":
            out["actions"].append(c)
        elif c["tier"] == "Watchlist":
            out["watchlist"].append(c)
        else:
            out["no_call"].append(c)

    # Select up to 3 actions under effort budget with category diversity
    selected, used, budget = [], set(), cfg.get("EFFORT_BUDGET", 8)
    for c in sorted(out["actions"], key=lambda x: x["score"], reverse=True):
        if c["category"] in used and len(selected) < 2:
            continue
        if sum(x["effort"] for x in selected) + c["effort"] > budget:
            out["backlog"].append({**c, "reason": "fails effort budget"})
            continue
        selected.append(c)
        used.add(c["category"])
        if len(selected) >= 3:
            break
    out["actions"] = selected

    # Pilot fallback (if no Actions)
    if len(out["actions"]) == 0 and len(final) > 0:
        pilot = sorted(final, key=lambda x: x["score"], reverse=True)[0]
        n_needed = pilot["min_n"]
        if pilot["metric"] in ("repeat_rate", "discount_rate"):
            p = pilot.get("baseline_rate", 0.15)
            delta = pilot.get("effect_floor", 0.02)
            n_needed = required_n_for_proportion(p if p > 0 else 0.15, delta if delta > 0 else 0.02, alpha=0.05, power=0.8)
        pilot_payload = {
            **pilot,
            "tier": "Pilot",
            "pilot_audience_fraction": cfg.get("PILOT_AUDIENCE_FRACTION", 0.2),
            "pilot_budget_cap": cfg.get("PILOT_BUDGET_CAP", 200.0),
            "n_needed": int(n_needed),
            "decision_rule": "Graduate to full rollout if CI excludes 0 or q ≤ α at 14 days, else rollback.",
            "confidence_label": "Low",
            "expected_range": [round((pilot.get("expected_$", 0) or 0) * 0.6, 2), round((pilot.get("expected_$", 0) or 0) * 1.3, 2)],
        }
        out["pilot_actions"] = [pilot_payload]

    # --- Build Backlog of near-misses (top 3 by score), after pilot to avoid duplicates ---
    backlog = []
    backlog.extend(out.get("backlog", []))  # keep budget/category spillover

    # Exclude the chosen pilot from backlog (if any)
    pilot_key = None
    if out.get("pilot_actions"):
        pilot0 = out["pilot_actions"][0]
        pilot_key = (pilot0.get("id"), pilot0.get("play_id"))

    near_misses = []
    for c in final:
        if c in out["actions"]:
            continue
        key = (c.get("id"), c.get("play_id"))
        if pilot_key and key == pilot_key:
            continue
        fail_cnt = len(c["failed"])
        if fail_cnt <= 1 or c["tier"] == "Watchlist":
            reason = ", ".join(c.get("reasons", [])) or "near miss"
            near_misses.append({**c, "reason": reason})

    seen_ids = {(b.get("id"), b.get("play_id")) for b in backlog}
    for c in sorted(near_misses, key=lambda x: x["score"], reverse=True):
        key = (c.get("id"), c.get("play_id"))
        if key in seen_ids:
            continue
        backlog.append({"id": c.get("id"), "play_id": c.get("play_id"), "title": c["title"], "reason": c.get("reason", "near miss")})
        seen_ids.add(key)
        if len(backlog) >= 3:
            break

    out["backlog"] = backlog
    return out

def write_actions_log(receipts_dir: str, actions: List[Dict[str, Any]]) -> None:
    log_path = Path(receipts_dir)/"actions_log.json"
    log = []
    if log_path.exists():
        log = json.loads(log_path.read_text())
    for a in actions:
        log.append({"play_id": a["play_id"], "title": a["title"]})
    write_json(str(log_path), log)
