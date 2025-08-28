from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np, pandas as pd, json
from .policy import load_policy, is_eligible
import datetime

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

def _load_actions_log(receipts_dir: str) -> list[dict]:
    p = Path(receipts_dir) / "actions_log.json"
    if not p.exists(): return []
    try: return json.loads(p.read_text())
    except Exception: return []

def _weeks_since_used(
    log: list[dict],
    play_id: str,
    variant_id: str | None,
    asof_date: datetime.date | None = None
) -> int | None:
    """
    Return how many whole ISO weeks since this play/variant was last used,
    relative to `asof_date` (default: today). 0 = same ISO week.
    """
    ts_list: list[datetime.date] = []
    for r in log:
        if r.get("play_id") != play_id: 
            continue
        if variant_id is not None and r.get("variant_id") != variant_id:
            continue
        try:
            d = datetime.datetime.fromisoformat(r["ts"].replace("Z","")).date()
            ts_list.append(d)
        except Exception:
            pass
    if not ts_list:
        return None
    last = max(ts_list)
    ref = asof_date or datetime.date.today()
    # ISO week difference (calendar weeks), not just 7-day buckets:
    last_iso_year, last_iso_week, _ = last.isocalendar()
    ref_iso_year, ref_iso_week, _ = ref.isocalendar()
    return (ref_iso_year - last_iso_year) * 52 + (ref_iso_week - last_iso_week)


# ---- Section partition helper (single source of truth for sections) ----
def _partition_candidates(final: list[dict], effort_budget: int = 8):
    """
    Sections:
      - Watchlist := failed ≥1 gate
      - Pool      := passed all gates
      - Top       := up to 3 from Pool under effort budget,
                     *one variant per play_id*,
                     and soft category diversity (try to include ≥2 categories).
      - Backlog   := remaining Pool items (all passed gates) with a defer reason
    """
    # 1) Classify
    watchlist = [c for c in final if c.get("failed")]          # failed at least one gate
    pool      = [c for c in final if not c.get("failed")]      # passed all gates

    # 2) Rank pool by score (desc)
    pool_sorted = sorted(pool, key=lambda x: x.get("score", 0), reverse=True)

    top: list[dict] = []
    used_cats: set   = set()
    used_plays: set  = set()
    used_effort: int = 0

    # --- First pass: enforce play-level uniqueness + category diversity ---
    for c in pool_sorted:
        if len(top) >= 3:
            break

        pid = c.get("play_id")
        cat = c.get("category")
        eff = int(c.get("effort", 2))

        # one-per-play family
        if pid in used_plays:
            c.setdefault("defer_reason", "another variant of this play selected")
            continue

        # soft category diversity: until we have ≥2 categories, avoid duplicates
        if cat in used_cats and len(used_cats) < 2:
            c.setdefault("defer_reason", "category diversity")
            continue

        # effort budget
        if used_effort + eff > effort_budget:
            c.setdefault("defer_reason", "effort budget")
            continue

        top.append(c)
        used_plays.add(pid)
        if cat:
            used_cats.add(cat)
        used_effort += eff

    # --- Second pass: if we still have <3, relax category diversity (but keep one-per-play & budget) ---
    if len(top) < 3:
        for c in pool_sorted:
            if len(top) >= 3:
                break

            pid = c.get("play_id")
            eff = int(c.get("effort", 2))

            # skip items already chosen
            if any((c.get("play_id"), c.get("variant_id")) == (t.get("play_id"), t.get("variant_id")) for t in top):
                continue

            # still enforce one-per-play family
            if pid in used_plays:
                c.setdefault("defer_reason", "another variant of this play selected")
                continue

            # budget
            if used_effort + eff > effort_budget:
                c.setdefault("defer_reason", "effort budget")
                continue

            top.append(c)
            used_plays.add(pid)
            used_effort += eff

    # 3) Build backlog: all pool items not chosen (they passed gates)
    chosen_keys = {(t.get("play_id"), t.get("variant_id")) for t in top}
    backlog: list[dict] = []
    for c in pool_sorted:
        key = (c.get("play_id"), c.get("variant_id"))
        if key not in chosen_keys:
            c.setdefault("defer_reason", "ranked below top actions")
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

def select_actions(g, aligned, cfg, playbooks_path: str, receipts_dir: str, policy_path: str | None = None) -> Dict[str, Any]:
    """
    1) Build base candidates (same as before)
    2) Gate + score (same as before)
    3) Expand into variants; apply cooldown + novelty penalty
    4) Partition with soft diversity + effort budget
    5) Pilot fallback if nothing passes
    """
    # --- load playbook + base candidates ---
    plays = {p["id"]: p for p in load_playbooks(playbooks_path)}
    base_cands = _compute_candidates(g, aligned, cfg)

    # FDR adjust (q-values) across base candidates only
    pvals = [c.get("p", np.nan) for c in base_cands]
    if any([not np.isnan(p) for p in pvals]):
        # your BH takes (pvals, alpha) -> (qvals, reject_mask_or_none)
        qvals, _ = benjamini_hochberg([p if not np.isnan(p) else 1.0 for p in pvals], cfg["FDR_ALPHA"])
        for i, c in enumerate(base_cands):
            c["q"] = qvals[i]

    # --- Gate + score + meta attach (unchanged from your prior flow) ---
    final: List[Dict[str, Any]] = []
    for c in base_cands:
        meta = plays.get(c["play_id"], {})
        # attach metadata (safe defaults)
        c["category"] = meta.get("category", "general")
        c["title"] = meta.get("title", c["play_id"])
        for k in [
            "do_this","targeting","channels","cadence","offer","copy_snippets","assets",
            "how_to_launch","success_criteria","risks_mitigations","owner_suggested",
            "time_to_set_up_minutes","holdout_plan"
        ]:
            c[k] = meta.get(k)
        c["effort"] = meta.get("effort", 2)
        c["risk"] = meta.get("risk", 2)

        cand = _gate_and_score(c, cfg)

        # confidence label + expected range (same logic as before)
        if len(cand["failed"]) == 0:
            cand["confidence_label"] = "High"
        elif ("min_n" in cand["passed"]) and ("significance" in cand["failed"]) and ("effect_floor" not in cand["failed"]) and ("financial_floor" not in cand["failed"]):
            cand["confidence_label"] = "Medium"
        else:
            cand["confidence_label"] = "Low"

        expv = float(cand.get("expected_$") or 0.0)
        cand["expected_range"] = [round(expv * 0.6, 2), round(expv * 1.3, 2)]

        final.append(cand)

    # ---------- INSERTED BLOCK STARTS HERE ----------
    # Expand into variant candidates + apply policy, cooldown, novelty
     # ---------- INSERTED BLOCK (corrected) ----------
# Expand into variant candidates + apply policy, cooldown, novelty
    try:
        from .policy import load_policy, is_eligible  # optional dependency
    except Exception:
        def load_policy(_=None): 
            return {"allow_free_shipping": True, "max_discount_pct": 15, "channel_caps": {"email_per_week": 2, "sms_per_week": 1}}
        def is_eligible(expr, policy): 
            return True

    # Anchor date for week-aware cooldown
    anchor_dt = None
    try:
        val = aligned.get("anchor")
        if val:
            anchor_dt = val.date() if hasattr(val, "date") else datetime.date.fromisoformat(str(val)[:10])
    except Exception:
        anchor_dt = None

    policy = load_policy(policy_path)
    log = _load_actions_log(receipts_dir)

    variant_cands: List[Dict[str, Any]] = []
    for cand in final:
        pmeta = plays.get(cand["play_id"], {})
        variants = pmeta.get("variants", [{"id": "base", "offer_type": "no_discount", "lift_multiplier": 1.0}])
        cooldown_weeks = int(pmeta.get("cooldown_weeks", 1))

        for v in variants:
            if not is_eligible(v.get("eligible_if", "True"), policy):
                continue

            vc = cand.copy()
            vc["variant_id"] = v.get("id", "base")
            vc["offer"] = {"type": v.get("offer_type"), "value": v.get("value")}

            # Expected $: base × lift − incentive_cost (concierge-simple)
            exp_base = float(vc.get("expected_$") or 0.0)
            lift_mult = float(v.get("lift_multiplier", 1.0))
            aov = float(aligned.get("recent_aov") or aligned.get("L28_aov") or 0.0)
            audience = int(vc.get("audience_size") or vc.get("n") or 0)

            cost_per_order = 0.0
            if v.get("cost_type") == "percent_of_aov" and aov > 0:
                cost_per_order = (float(v.get("cost_value", 0.0)) / 100.0) * aov
            elif v.get("cost_type") == "flat_per_order":
                cost_per_order = float(v.get("cost_value", 0.0))
            incentive_cost = cost_per_order * audience

            vc["expected_$"] = max(0.0, exp_base * lift_mult - incentive_cost)

            # Novelty & cooldown (week-aware)
            weeks_variant = _weeks_since_used(log, vc["play_id"], vc["variant_id"], asof_date=anchor_dt)
            weeks_family  = _weeks_since_used(log, vc["play_id"], None,           asof_date=anchor_dt)

            penalty = 0.0
            if weeks_variant == 0: penalty = 0.25
            elif weeks_variant == 1: penalty = 0.15
            elif weeks_variant == 2: penalty = 0.05

            # Cooldown applies only across weeks (>=1), not within same week (0)
            if (weeks_family is not None) and (weeks_family >= 1) and (weeks_family < cooldown_weeks):
                # skip this family this week due to cooldown
                continue

            vc["score"] = (vc.get("score") or 0.0) * (1.0 - penalty)
            variant_cands.append(vc)

    # If cooldown filtered everything, fall back to passed candidates (ignore cooldown)
    if not variant_cands:
        variant_cands = []
        for cand in final:
            if cand.get("failed"):   # keep only passes
                continue
            vc = cand.copy()
            vc.setdefault("variant_id", "base")
            variant_cands.append(vc)

    # Use variants for selection
    finals_for_selection = variant_cands if variant_cands else final
    # ---------- INSERTED BLOCK END ----------

    # ---------- INSERTED BLOCK ENDS HERE ----------

    # --- Partition & select (soft diversity + effort budget) ---
    budget = cfg.get("EFFORT_BUDGET", 8)
    top_actions, backlog, watchlist = _partition_candidates(finals_for_selection, effort_budget=budget)

    out = {
        "actions": top_actions,
        "watchlist": [c for c in final if c.get("failed")],  # failed ≥1 gate (no attachments)
        "no_call": [],
        "backlog": [],
        "pilot_actions": [],
    }

    # Backlog: passed all gates but deferred (keep reason)
    for b in backlog:
        out["backlog"].append({
            **b,
            "reason": b.get("defer_reason", "ranked below top actions"),
        })

    # Pilot fallback (if nothing made it into Top Actions)
    if len(out["actions"]) == 0 and len(final) > 0:
        pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
        n_needed = pilot.get("min_n", 0)
        if pilot.get("metric") in ("repeat_rate", "discount_rate"):
            p = pilot.get("baseline_rate", 0.15) or 0.15
            delta = pilot.get("effect_floor", 0.02) or 0.02
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


def write_actions_log(receipts_dir: str, actions: list[dict]) -> None:
    log_path = Path(receipts_dir) / "actions_log.json"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log = []
    if log_path.exists():
        try: log = json.loads(log_path.read_text())
        except Exception: log = []
    for a in actions:
        log.append({
            "ts": now,
            "play_id": a.get("play_id"),
            "variant_id": a.get("variant_id"),
            "title": a.get("title"),
        })
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

# --- Evidence builder (drop-in) --- #
def _pct(x, digits=1):
    return f"{x*100:.{digits}f}%" if (x is not None) else "—"

def _money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"

def _num(x):
    try:
        return f"{int(x)}"
    except Exception:
        return "—"

def _safe_get(al, path, default=None):
    cur = al
    try:
        for k in path:
            cur = cur[k]
        return cur
    except Exception:
        return default

def _receipt_discount_hygiene(al, a):
    dr1 = _safe_get(al, ["L28","discount_rate"])
    dr0 = _safe_get(al, ["L28","prior","discount_rate"])
    ddr = _safe_get(al, ["L28","delta","discount_rate"])
    p   = _safe_get(al, ["L28","p","discount_rate"])
    sig = _safe_get(al, ["L28","sig","discount_rate"])
    aov_delta = _safe_get(al, ["L28","delta","aov"])
    est = _money(a.get("expected_$"))
    msg = []
    if dr1 is not None and dr0 is not None:
        msg.append(f"Discounted-order share rose from {_pct(dr0)} → {_pct(dr1)} ({_pct(ddr)} vs prior){' [significant]' if sig else ''}.")
    if aov_delta is not None:
        msg.append(f"AOV {_pct(aov_delta)} vs prior (flat/down suggests margin leakage).")
    msg.append(f"Guardrail expected to recover ≈ {est}.")
    return " ".join(msg)

def _receipt_winback(al, a):
    rr1 = _safe_get(al, ["L28","repeat_share"])
    rr0 = _safe_get(al, ["L28","prior","repeat_share"])
    drr = _safe_get(al, ["L28","delta","repeat_share"])
    p   = _safe_get(al, ["L28","p","repeat_share"])
    sig = _safe_get(al, ["L28","sig","repeat_share"])
    idn = _safe_get(al, ["L28","meta","identified_recent"], 0)
    est = _money(a.get("expected_$"))
    parts = []
    if rr1 is not None and rr0 is not None:
        parts.append(f"Repeat share {_pct(rr1)} (was {_pct(rr0)}; {_pct(drr)} vs prior){' [significant]' if sig else ''}.")
    parts.append(f"Identified customers this period: {_num(idn)}.")
    parts.append(f"Win-back cohort expected value ≈ {est}.")
    return " ".join(parts)

def _receipt_bestseller(al, a):
    aov_d = _safe_get(al, ["L28","delta","aov"])
    orders_d = _safe_get(al, ["L28","delta","orders"])
    est = _money(a.get("expected_$"))
    msg = []
    if aov_d is not None:
        msg.append(f"AOV {_pct(aov_d)} vs prior; bundling/hero placement aims to extend this lift.")
    if orders_d is not None:
        msg.append(f"Orders {_pct(orders_d)}; amplifying top seller targets attach rate.")
    msg.append(f"Expected impact ≈ {est}.")
    return " ".join(msg)

def _receipt_dormant(al, a):
    rr0 = _safe_get(al, ["L28","prior","repeat_share"])
    rr1 = _safe_get(al, ["L28","repeat_share"])
    est = _money(a.get("expected_$"))
    msg = []
    if rr1 is not None and rr0 is not None:
        msg.append(f"Store repeat share {_pct(rr1)} vs {_pct(rr0)} prior; reactivating multi-buyers should lift frequency.")
    msg.append(f"Expected impact ≈ {est}.")
    return " ".join(msg)

def evidence_for_action(action: dict, aligned: dict) -> list[str]:
    """Return a few concise, numeric reasons for this specific action."""
    pid = (action.get("play_id") or "").lower()
    if "discount_hygiene" in pid:
        return [_receipt_discount_hygiene(aligned, action)]
    if "winback" in pid:
        return [_receipt_winback(aligned, action)]
    if "bestseller" in pid or "amplify" in pid:
        return [_receipt_bestseller(aligned, action)]
    if "dormant" in pid:
        return [_receipt_dormant(aligned, action)]
    # fallback: use rationale/effect
    eff = action.get("effect_abs")
    return [action.get("rationale") or f"Effect delta {_pct(eff)} vs prior; expected ≈ {_money(action.get('expected_$'))}."]

def build_receipts(aligned: dict, actions_bundle: dict) -> list[str]:
    """
    Take selected actions (and pilot if any) and produce 3–5 'why this will work' bullets.
    """
    out = []
    # Top actions first
    for a in actions_bundle.get("actions", []):
        out += evidence_for_action(a, aligned)
    # If still sparse, include pilot rationale
    for p in actions_bundle.get("pilot_actions", []):
        out += evidence_for_action(p, aligned)
    # Keep it tight
    uniq = []
    seen = set()
    for s in out:
        k = s.strip()
        if k and k not in seen:
            uniq.append(k); seen.add(k)
        if len(uniq) >= 5:
            break
    return uniq
# --- end evidence builder --- #
