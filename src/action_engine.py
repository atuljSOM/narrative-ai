from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np, pandas as pd, json
from .policy import load_policy, is_eligible
import datetime
from enum import Enum

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
from .utils import get_interaction_factors
from .utils import subscription_threshold_for_product, categorize_product
from .features import compute_repeat_curve, build_g_items

class ActionStatus(Enum):
    """Status tracking for recommended actions"""
    PENDING = "pending"
    IMPLEMENTED = "implemented"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class ActionTracker:
    """
    Tracks implementation and results of recommended actions.
    Stores historical performance to improve future predictions.
    """
    
    def __init__(self, receipts_dir: str):
        self.receipts_dir = Path(receipts_dir)
        self.outcomes_file = self.receipts_dir / "action_outcomes.json"
        self.performance_file = self.receipts_dir / "prediction_performance.json"
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        
    def track_action(
        self,
        action_id: str,
        play_id: str,
        variant_id: str = "base",
        status: ActionStatus = ActionStatus.PENDING,
        predicted_revenue: float = 0.0,
        predicted_effect: float = 0.0,
        confidence_score: float = 0.0,
        timestamp: datetime.datetime = None
    ) -> Dict[str, Any]:
        """Initialize tracking for a new action."""
        timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
        
        outcome = {
            "action_id": action_id,
            "play_id": play_id,
            "variant_id": variant_id,
            "status": status.value,
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat(),
            "predicted": {
                "revenue": float(predicted_revenue),
                "effect": float(predicted_effect),
                "confidence": float(confidence_score),
                # Monthly plan: measure over ~28 days
                "expected_complete_by": (timestamp + datetime.timedelta(days=28)).isoformat()
            },
            "actual": {
                "revenue": None,
                "effect": None,
                "implemented_at": None,
                "completed_at": None,
                "notes": None
            },
            "validation": {
                "data_quality_at_recommendation": None,
                "implementation_verified": False,
                "results_verified": False
            }
        }
        
        # Load existing outcomes
        outcomes = self._load_outcomes()
        outcomes[action_id] = outcome
        self._save_outcomes(outcomes)
        
        return outcome
    
    def update_implementation(
        self,
        action_id: str,
        implemented: bool,
        implementation_notes: str = None,
        channels_used: List[str] = None,
        audience_size_actual: int = None
    ) -> Dict[str, Any]:
        """Track when an action is actually implemented."""
        outcomes = self._load_outcomes()
        
        if action_id not in outcomes:
            raise ValueError(f"Action {action_id} not found in tracker")
        
        outcome = outcomes[action_id]
        now = datetime.datetime.now(datetime.timezone.utc)
        
        if implemented:
            outcome["status"] = ActionStatus.IN_PROGRESS.value
            outcome["actual"]["implemented_at"] = now.isoformat()
            outcome["actual"]["implementation_notes"] = implementation_notes
            outcome["actual"]["channels_used"] = channels_used
            outcome["actual"]["audience_size"] = audience_size_actual
            outcome["validation"]["implementation_verified"] = True
        else:
            outcome["status"] = ActionStatus.SKIPPED.value
            outcome["actual"]["skipped_reason"] = implementation_notes
        
        outcome["updated_at"] = now.isoformat()
        
        outcomes[action_id] = outcome
        self._save_outcomes(outcomes)
        
        return outcome
    
    def track_results(
        self,
        action_id: str,
        actual_revenue: float,
        actual_orders: int = None,
        actual_conversion_rate: float = None,
        measurement_period_days: int = 14,
        notes: str = None
    ) -> Dict[str, Any]:
        """Track actual results after measurement period."""
        outcomes = self._load_outcomes()
        
        if action_id not in outcomes:
            raise ValueError(f"Action {action_id} not found in tracker")
        
        outcome = outcomes[action_id]
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Calculate actual effect if we have conversion data
        actual_effect = None
        if actual_conversion_rate is not None and outcome["predicted"]["effect"]:
            # This assumes effect was measured as a rate change
            baseline = outcome["predicted"].get("baseline_rate", 0.15)
            actual_effect = actual_conversion_rate - baseline
        
        outcome["status"] = ActionStatus.COMPLETED.value
        outcome["actual"]["revenue"] = float(actual_revenue)
        outcome["actual"]["orders"] = actual_orders
        outcome["actual"]["conversion_rate"] = actual_conversion_rate
        outcome["actual"]["effect"] = actual_effect
        outcome["actual"]["measurement_period_days"] = measurement_period_days
        outcome["actual"]["completed_at"] = now.isoformat()
        outcome["actual"]["notes"] = notes
        outcome["validation"]["results_verified"] = True
        
        # Calculate accuracy metrics
        predicted_rev = outcome["predicted"]["revenue"]
        if predicted_rev > 0:
            accuracy_pct = (actual_revenue / predicted_rev) * 100
            outcome["validation"]["revenue_accuracy_pct"] = round(accuracy_pct, 1)
            outcome["validation"]["prediction_quality"] = self._rate_prediction(accuracy_pct)
        
        outcome["updated_at"] = now.isoformat()
        
        outcomes[action_id] = outcome
        self._save_outcomes(outcomes)
        
        # Update performance tracking
        self._update_performance_metrics(outcome)
        
        return outcome
    
    def get_action_status(self, action_id: str) -> Dict[str, Any]:
        """Get current status of an action."""
        outcomes = self._load_outcomes()
        return outcomes.get(action_id)
    
    def get_performance_summary(self, play_id: str = None) -> Dict[str, Any]:
        """Get historical performance summary, optionally filtered by play_id."""
        outcomes = self._load_outcomes()
        perf = self._load_performance()
        
        # Filter completed actions
        completed = [
            o for o in outcomes.values()
            if o["status"] == ActionStatus.COMPLETED.value
            and (play_id is None or o["play_id"] == play_id)
        ]
        
        if not completed:
            return {
                "n_completed": 0,
                "message": "No completed actions to analyze"
            }
        
        # Calculate aggregate metrics
        total_predicted = sum(o["predicted"]["revenue"] for o in completed)
        total_actual = sum(o["actual"]["revenue"] for o in completed)
        
        # Accuracy distribution
        accuracies = [
            o["validation"].get("revenue_accuracy_pct", 0)
            for o in completed
            if o["validation"].get("revenue_accuracy_pct")
        ]
        
        summary = {
            "n_completed": len(completed),
            "total_predicted_revenue": round(total_predicted, 2),
            "total_actual_revenue": round(total_actual, 2),
            "aggregate_accuracy_pct": round((total_actual / total_predicted * 100) if total_predicted > 0 else 0, 1),
            "median_accuracy_pct": round(float(np.median(accuracies)), 1) if accuracies else None,
            "actions_over_performed": sum(1 for a in accuracies if a > 110),
            "actions_under_performed": sum(1 for a in accuracies if a < 90),
            "actions_on_target": sum(1 for a in accuracies if 90 <= a <= 110),
            "play_id": play_id
        }
        
        # Add play-specific performance if available
        if play_id and play_id in perf:
            summary["play_performance"] = perf[play_id]
        
        return summary
    
    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get all actions awaiting implementation or results."""
        outcomes = self._load_outcomes()
        pending = [
            o for o in outcomes.values()
            if o["status"] in [ActionStatus.PENDING.value, ActionStatus.IN_PROGRESS.value]
        ]
        
        # Sort by age (oldest first)
        pending.sort(key=lambda x: x["created_at"])
        
        # Add days_waiting for each
        now = datetime.datetime.now(datetime.timezone.utc)
        for p in pending:
            created = datetime.datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
            p["days_waiting"] = (now - created).days
            
            # Flag if overdue
            if p["status"] == ActionStatus.IN_PROGRESS.value:
                expected = datetime.datetime.fromisoformat(
                    p["predicted"]["expected_complete_by"].replace("Z", "+00:00")
                )
                p["is_overdue"] = now > expected
        
        return pending
    
    def generate_weekly_performance_report(self) -> str:
        """Generate a markdown report of prediction performance."""
        summary = self.get_performance_summary()
        pending = self.get_pending_actions()
        
        report = f"""# Aura Prediction Performance Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overall Accuracy
- **Actions Completed**: {summary['n_completed']}
- **Aggregate Accuracy**: {summary['aggregate_accuracy_pct']}%
- **Median Accuracy**: {summary.get('median_accuracy_pct', 'N/A')}%

## Revenue Impact
- **Total Predicted**: ${summary['total_predicted_revenue']:,.2f}
- **Total Actual**: ${summary['total_actual_revenue']:,.2f}
- **Variance**: ${summary['total_actual_revenue'] - summary['total_predicted_revenue']:+,.2f}

## Prediction Quality
- **Over-performed (>110%)**: {summary['actions_over_performed']} actions
- **On Target (90-110%)**: {summary['actions_on_target']} actions  
- **Under-performed (<90%)**: {summary['actions_under_performed']} actions

## Pending Actions
"""
        
        if pending:
            report += f"**{len(pending)} actions awaiting results:**\n\n"
            for p in pending[:5]:  # Show top 5
                status_emoji = "⏳" if p["status"] == ActionStatus.PENDING.value else "🚀"
                overdue = " ⚠️ OVERDUE" if p.get("is_overdue") else ""
                report += f"- {status_emoji} {p['play_id']} (Day {p['days_waiting']}){overdue}\n"
        else:
            report += "*No pending actions*\n"
        
        report += "\n---\n*Use these insights to calibrate future predictions*"
        
        return report
    
    def _load_outcomes(self) -> Dict[str, Dict[str, Any]]:
        """Load action outcomes from disk."""
        if self.outcomes_file.exists():
            try:
                with open(self.outcomes_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_outcomes(self, outcomes: Dict[str, Dict[str, Any]]):
        """Save action outcomes to disk."""
        write_json(str(self.outcomes_file), outcomes)
    
    def _load_performance(self) -> Dict[str, Any]:
        """Load performance metrics from disk."""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_performance(self, performance: Dict[str, Any]):
        """Save performance metrics to disk."""
        write_json(str(self.performance_file), performance)
    
    def _update_performance_metrics(self, outcome: Dict[str, Any]):
        """Update aggregated performance metrics for a play type."""
        perf = self._load_performance()
        play_id = outcome["play_id"]
        
        if play_id not in perf:
            perf[play_id] = {
                "n": 0,
                "total_predicted": 0.0,
                "total_actual": 0.0,
                "accuracies": [],
                "last_updated": None
            }
        
        p = perf[play_id]
        p["n"] += 1
        p["total_predicted"] += outcome["predicted"]["revenue"]
        p["total_actual"] += outcome["actual"]["revenue"]
        
        if outcome["validation"].get("revenue_accuracy_pct"):
            p["accuracies"].append(outcome["validation"]["revenue_accuracy_pct"])
        
        p["avg_accuracy"] = round(float(np.mean(p["accuracies"])), 1) if p["accuracies"] else None
        p["median_accuracy"] = round(float(np.median(p["accuracies"])), 1) if p["accuracies"] else None
        p["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        self._save_performance(perf)
    
    def _rate_prediction(self, accuracy_pct: float) -> str:
        """Rate prediction quality based on accuracy."""
        if accuracy_pct >= 90 and accuracy_pct <= 110:
            return "excellent"
        elif accuracy_pct >= 75 and accuracy_pct <= 125:
            return "good"
        elif accuracy_pct >= 50 and accuracy_pct <= 150:
            return "fair"
        else:
            return "poor"

# Add this to the existing _load_actions_log function
def _load_actions_log(receipts_dir: str) -> list[dict]:
    p = Path(receipts_dir) / "actions_log.json"
    if not p.exists(): return []
    try: return json.loads(p.read_text())
    except Exception: return []

# Primary implementation for action selection (inventory-aware)
def _select_actions_impl(
    g,
    aligned,
    cfg,
    playbooks_path: str,
    receipts_dir: str,
    policy_path: str | None = None,
    inventory_metrics: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Select top actions, backlog, and pilot based on candidates and policy.

    This consolidates gating, scoring, variant expansion, cooldowns, inventory-awareness,
    overlap adjustments, and interaction effects into a single implementation.
    """
    # Load playbooks and build quick lookup by id
    try:
        plays_list = load_playbooks(playbooks_path)
    except Exception:
        plays_list = []
    plays: Dict[str, Any] = {str(p.get("id")): p for p in plays_list if isinstance(p, dict)}

    # Compute base candidates from recent performance signals
    base_cands: List[Dict[str, Any]] = _compute_candidates(g, aligned, cfg)

    # LTV signal (non-gating): store-level and top-decile for receipts and tie-breakers
    ltv_info = compute_repeat_curve(g, horizon_days=[60, 90]) if g is not None else {"store": {}, "per_customer": []}
    store_ltv90 = float(((ltv_info or {}).get("store", {}) or {}).get(90, {}).get("ltv", 0.0) or 0.0)
    try:
        arr = np.array([float(x.get("ltv90", 0.0) or 0.0) for x in (ltv_info.get("per_customer", []) or [])], dtype=float)
        ltv90_p90 = float(np.quantile(arr, 0.9)) if arr.size > 0 else 0.0
    except Exception:
        ltv90_p90 = 0.0

    # FDR adjust (q-values) across base candidates only
    pvals = [c.get("p", np.nan) for c in base_cands]
    if any([not np.isnan(p) for p in pvals]):
        qvals, _ = benjamini_hochberg([p if not np.isnan(p) else 1.0 for p in pvals], cfg["FDR_ALPHA"])
        for i, c in enumerate(base_cands):
            c["q"] = qvals[i]

    # Gate + score + attach playbook metadata
    final: List[Dict[str, Any]] = []
    for c in base_cands:
        meta = plays.get(c.get("play_id"), {}) or {}
        c = c.copy()
        c["category"] = meta.get("category", "general")
        c["title"] = meta.get("title", c.get("play_id"))
        for k in [
            "do_this","targeting","channels","cadence","offer","copy_snippets","assets",
            "how_to_launch","success_criteria","risks_mitigations","owner_suggested",
            "time_to_set_up_minutes","holdout_plan"
        ]:
            if k in meta:
                c[k] = meta.get(k)

        cand = _gate_and_score(c, cfg)

        # Confidence label + expected range
        if len(cand["failed"]) == 0:
            cand["confidence_label"] = "High"
        elif ("min_n" in cand["passed"]) and ("significance" in cand["failed"]) and ("effect_floor" not in cand["failed"]) and ("financial_floor" not in cand["failed"]):
            cand["confidence_label"] = "Medium"
        else:
            cand["confidence_label"] = "Low"

        expv = float(cand.get("expected_$") or 0.0)
        cand["expected_range"] = [round(expv * 0.6, 2), round(expv * 1.3, 2)]

        # attach LTV estimates (non-gating)
        cand["audience_ltv90"] = store_ltv90
        cand["ltv90_top_decile"] = ltv90_p90
        final.append(cand)

    # Variant expansion + policy eligibility + cooldown/novelty + inventory-aware adjustments
    try:
        from .policy import load_policy as _lp, is_eligible as _ie  # optional dependency
        _load_pol = _lp
        _is_el = _ie
    except Exception:
        def _load_pol(_=None):
            return {"allow_free_shipping": True, "max_discount_pct": 15, "channel_caps": {"email_per_week": 2, "sms_per_week": 1}}
        def _is_el(expr, policy):
            return True

    # Anchor date for week-aware cooldown
    anchor_dt = None
    try:
        val = aligned.get("anchor")
        if val:
            anchor_dt = val.date() if hasattr(val, "date") else datetime.date.fromisoformat(str(val)[:10])
    except Exception:
        anchor_dt = None

    policy = _load_pol(policy_path)
    log = _load_actions_log(receipts_dir)

    variant_cands: List[Dict[str, Any]] = []

    # Inventory helpers (soft enforcement by default)
    inv_df = None
    try:
        inv_df = inventory_metrics.copy() if inventory_metrics is not None else None
    except Exception:
        inv_df = None

    def _inv_summary():
        if inv_df is None or inv_df.empty:
            return None
        tmp = {}
        try:
            tmp['in_stock_ratio14'] = float((inv_df['cover_days'] >= 14).mean()) if 'cover_days' in inv_df.columns else 1.0
            tmp['cover_min'] = float(inv_df['cover_days'].min()) if 'cover_days' in inv_df.columns else float('inf')
            tmp['cover_p25'] = float(inv_df['cover_days'].quantile(0.25)) if 'cover_days' in inv_df.columns else float('inf')
            tf = inv_df.get('trust_factor')
            tmp['trust_mean'] = float(tf.mean()) if tf is not None else 1.0
        except Exception:
            tmp = None
        return tmp

    inv_sum = _inv_summary()
    inv_mode = str(cfg.get('INVENTORY_ENFORCEMENT_MODE','soft') or 'soft').lower()
    cover_map = cfg.get('INVENTORY_MIN_COVER_DAYS') or {}
    default_cover = int(float((cover_map.get('default') or 21))) if cover_map else 21

    def _targeted_skus_for_play(vc: Dict[str, Any]) -> list[str]:
        try:
            play_id = str(vc.get('play_id') or '').lower()
            if inv_df is None or inv_df.empty:
                return []
            if 'lineitem_any' not in g.columns and 'SKU' not in g.columns:
                return []
            maxd = pd.to_datetime(g["Created at"]).max()
            start = maxd - pd.Timedelta(days=int(aligned.get("window_days") or 28) - 1)
            gg = g[g['Created at'] >= start].copy()
            def top_n_from(col: str, n: int = 3):
                ser = gg[col].astype(str)
                units = pd.to_numeric(gg.get('Lineitem quantity', pd.Series(1, index=gg.index)), errors='coerce').fillna(1)
                cnt = units.groupby(ser).sum().sort_values(ascending=False)
                return [str(x) for x in cnt.head(n).index.tolist()]
            if 'bestseller' in play_id or 'amplify' in play_id:
                # Prefer g_items for robust per-product targeting
                try:
                    gi = build_g_items(gg)
                    if gi is not None and not gi.empty:
                        rank = gi.groupby('product_key')['orders_product'].sum().sort_values(ascending=False)
                        return [str(x) for x in rank.head(3).index.tolist()]
                except Exception:
                    pass
                col = 'sku' if 'sku' in inv_df.columns else 'lineitem_any'
                return top_n_from(col, 3)
            if 'subscription_nudge' in play_id:
                start90 = maxd - pd.Timedelta(days=90)
                ww = g[g['Created at'] >= start90].copy()
                try:
                    rep = build_g_items(ww)
                    if rep is not None and not rep.empty:
                        use_col = 'product_key_base' if 'product_key_base' in rep.columns and bool(cfg.get('FEATURES_PRODUCT_NORMALIZATION', False)) else 'product_key'
                        subs = rep[rep['orders_product'] >= 3][use_col].astype(str).unique().tolist()
                        return subs[:5]
                except Exception:
                    pass
                if 'lineitem_any' in ww.columns:
                    rep = (ww.groupby(['customer_id','lineitem_any'])['Name']
                             .nunique().reset_index(name='orders_product'))
                    subs = rep[rep['orders_product'] >= 3]['lineitem_any'].astype(str).unique().tolist()
                    return subs[:5]
                return []
            if 'sample_to_full' in play_id:
                tokens = gg.get('lineitem_any', pd.Series([], dtype=str)).astype(str).str.lower()
                non_sample = gg.loc[~tokens.str.contains(r"sample|travel|mini|trial", regex=True, na=False), 'lineitem_any']
                return list(pd.Series(non_sample).astype(str).value_counts().head(3).index)
            if 'empty_bottle' in play_id:
                names = gg.get('lineitem_any', pd.Series([], dtype=str)).astype(str).str.lower()
                mask = names.str.contains(r"30ml|1 oz|1oz|50ml|1.7 oz|1.7oz|100ml|3.4 oz|3.4oz", regex=True, na=False)
                return list(gg.loc[mask, 'lineitem_any'].astype(str).value_counts().head(5).index)
        except Exception:
            return []
        return []

    def _apply_inventory_to_variant(vc: Dict[str, Any]):
        if inv_df is None or inv_df.empty:
            return
        play_id = str(vc.get('play_id') or '').lower()
        min_cover = int(float(cover_map.get(play_id, default_cover))) if cover_map else default_cover
        aov = float(aligned.get('recent_aov') or aligned.get('L28_aov') or 0.0)
        gm = float(cfg.get('GROSS_MARGIN', 0.70) or 0.70)
        expected = float(vc.get('expected_$') or 0.0)
        denom = max(aov * gm, 1e-6)
        required_units = expected / denom
        skus = _targeted_skus_for_play(vc)
        if not skus:
            return
        rows = inv_df[inv_df['sku'].astype(str).isin([str(s) for s in skus])].copy()
        if rows.empty:
            return
        available_cap = float(rows.get('available_net', pd.Series(dtype=float)).sum())
        cover_min = float(rows.get('cover_days', pd.Series([float('inf')])).min())
        trust_mean = float(rows.get('trust_factor', pd.Series([1.0])).mean())
        fulfillment = 1.0
        if required_units > 0:
            fulfillment = min(1.0, available_cap / max(1.0, required_units))
        fulfillment *= max(0.5, min(1.0, trust_mean))
        vc['inv_fulfillment'] = round(fulfillment, 2)
        vc['inv_cover_min'] = round(cover_min, 1)
        vc['inv_skus_count'] = int(len(rows))
        vc['expected_$'] = max(0.0, expected * fulfillment)
        if cover_min < min_cover:
            if inv_mode == 'hard':
                vc['__skip_due_inventory__'] = True
            else:
                vc.setdefault('notes', []).append(f"Low coverage for targeted SKUs (min≈{cover_min:.0f}d < {min_cover}d)")
                vc['score'] = (vc.get('score') or 0.0) * 0.9

    for cand in final:
        pmeta = plays.get(cand.get("play_id"), {})
        variants = pmeta.get("variants", [{"id": "base", "offer_type": "no_discount", "lift_multiplier": 1.0}])
        cooldown_weeks = int(pmeta.get("cooldown_weeks", 1))

        for v in variants:
            if not _is_el(v.get("eligible_if", "True"), policy):
                continue

            vc = cand.copy()
            vc.setdefault("variant_id", v.get("id", "base"))
            vc["variant_id"] = v.get("id", vc["variant_id"]) or "base"
            # Expected impact adjustment by variant lift
            lift = float(v.get("lift_multiplier", 1.0) or 1.0)
            vc["expected_$"] = float(vc.get("expected_$") or 0.0) * lift
            # Soft monthly scaling fallback if needed
            try:
                if (aligned.get('window_days') or 28) < 28:
                    vc["expected_$"] *= (28.0 / max(1.0, float(aligned.get('window_days') or 28)))
            except Exception:
                vc["expected_$"] *= 4.0

            # High-LTV nudge toward no-discount
            try:
                aud_ltv = float(cand.get("audience_ltv90") or 0.0)
                top_dec = float(cand.get("ltv90_top_decile") or 0.0)
                high_ltv = (aud_ltv >= top_dec) and (top_dec > 0)
            except Exception:
                high_ltv = False
            offer_type = str(v.get("offer_type", "")).lower()
            if high_ltv and ("discount" in offer_type or "percent_of_aov" in str(v.get("cost_type",""))):
                vc["expected_$"] *= 0.97
            elif high_ltv and (offer_type == "no_discount"):
                vc["expected_$"] *= 1.03

            # Novelty & cooldown (week-aware)
            weeks_variant = _weeks_since_used(log, vc["play_id"], vc["variant_id"], asof_date=anchor_dt)
            weeks_family  = _weeks_since_used(log, vc["play_id"], None,           asof_date=anchor_dt)

            penalty = 0.0
            if weeks_variant == 0: penalty = 0.25
            elif weeks_variant == 1: penalty = 0.15
            elif weeks_variant == 2: penalty = 0.05

            # Cooldown applies across weeks (>=1)
            if (weeks_family is not None) and (weeks_family >= 1) and (weeks_family < cooldown_weeks):
                continue

            # Inventory-aware adjustments (soft by default)
            if inv_sum is not None:
                play_id = str(vc.get('play_id') or '').lower()
                min_cover = int(float(cover_map.get(play_id, default_cover))) if cover_map else default_cover
                cover_health = inv_sum.get('cover_min', float('inf'))
                trust_mean = float(inv_sum.get('trust_mean', 1.0))
                vc["expected_$"] *= trust_mean
                vc['inv_trust'] = round(trust_mean, 2)
                if 'discount_hygiene' in play_id:
                    v_id = str(vc.get('variant_id') or '').lower()
                    if v_id.startswith('targeted'):
                        skus = _targeted_skus_for_play(vc)
                        if skus:
                            rows = inv_df[inv_df['sku'].astype(str).isin([str(s) for s in skus])].copy()
                            if not rows.empty and 'cover_days' in rows.columns:
                                total = int(len(rows)) or 1
                                low = int((rows['cover_days'] < 14).sum())
                                keep_ratio = max(0.0, min(1.0, (total - low) / total))
                                vc['expected_$'] *= keep_ratio
                                vc.setdefault('notes', []).append(f"Discount scope: product-specific — excluded {low}/{total} low-stock SKUs (≥14d cover required)")
                    else:
                        in_stock_ratio = float(inv_sum.get('in_stock_ratio14', 1.0))
                        vc['inv_in_stock_ratio14'] = round(in_stock_ratio, 2)
                        vc["expected_$"] *= in_stock_ratio
                        vc.setdefault('notes', []).append(f"Discount scope: sitewide — in-stock≥14d ratio≈{in_stock_ratio:.0%}")
                if cover_health < min_cover:
                    if inv_mode == 'hard':
                        continue
                    vc.setdefault('notes', []).append(f"Inventory cover low (min≈{cover_health:.0f}d < {min_cover}d)")
                    vc['score'] = (vc.get('score') or 0.0) * 0.9
                    vc['inv_cover_min'] = float(cover_health)

            _apply_inventory_to_variant(vc)
            if vc.get('__skip_due_inventory__'):
                continue

            vc["score"] = (vc.get("score") or 0.0) * (1.0 - penalty)
            variant_cands.append(vc)

    # If cooldown filtered everything, fall back to base candidates (ignore cooldown)
    if not variant_cands:
        for cand in final:
            if cand.get("failed"):
                continue
            vc = cand.copy()
            vc.setdefault("variant_id", "base")
            variant_cands.append(vc)

    finals_for_selection = variant_cands if variant_cands else final

    # Partition & select (soft diversity + effort budget)
    budget = cfg.get("EFFORT_BUDGET", 8)
    top_actions, backlog, watchlist = _partition_candidates(finals_for_selection, effort_budget=budget)

    out = {
        "actions": top_actions,
        "watchlist": [c for c in final if c.get("failed")],
        "no_call": [],
        "backlog": [],
        "pilot_actions": [],
    }

    # Backlog: passed all gates but deferred
    for b in backlog:
        out["backlog"].append({
            **b,
            "reason": b.get("defer_reason", "ranked below top actions"),
        })

    # Confidence Mode: conservative (default), aggressive, learning
    mode = str((cfg or {}).get("CONFIDENCE_MODE", "conservative")).strip().lower()

    def _mk_pilot(pilot: dict, note: str) -> dict:
        n_needed = pilot.get("min_n", 0)
        if pilot.get("metric") in ("repeat_rate", "discount_rate"):
            p = pilot.get("baseline_rate", 0.15) or 0.15
            delta = pilot.get("effect_floor", 0.02) or 0.02
            n_needed = required_n_for_proportion(p, delta, alpha=0.05, power=0.8)
        exp = float(pilot.get("expected_$", 0.0) or 0.0)
        return {
            **pilot,
            "tier": "Pilot",
            "pilot_audience_fraction": cfg.get("PILOT_AUDIENCE_FRACTION", 0.2),
            "pilot_budget_cap": cfg.get("PILOT_BUDGET_CAP", 200.0),
            "n_needed": int(n_needed),
            "decision_rule": "Graduate if CI excludes 0 or q ≤ α at 28 days; else rollback.",
            "confidence_label": pilot.get("confidence_label", "Low"),
            "expected_range": [round(exp * 0.6, 2), round(exp * 1.3, 2)],
            "notes": (pilot.get("notes") or []) + [note],
        }

    finals_pool = finals_for_selection if finals_for_selection else final

    if mode == 'conservative':
        # Pilot fallback only if no actions
        if len(out["actions"]) == 0 and len(final) > 0:
            pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
            out["pilot_actions"] = [_mk_pilot(pilot, "Conservative fallback pilot")]
    elif mode == 'aggressive':
        # Include up to 2 medium-confidence items (fails significance only; min_n + effect + financial pass)
        meds = []
        for c in finals_pool:
            failed = set(c.get('failed', [])); passed = set(c.get('passed', []))
            if ('significance' in failed) and ('min_n' in passed) and ('effect_floor' not in failed) and ('financial_floor' not in failed):
                meds.append(c)
        meds = sorted(meds, key=lambda x: x.get('score', 0), reverse=True)[:2]
        out["pilot_actions"] = [_mk_pilot(m, "Aggressive mode: medium-confidence (fails significance only)") for m in meds]
        if not out["pilot_actions"] and len(out["actions"]) == 0 and len(final) > 0:
            pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
            out["pilot_actions"] = [_mk_pilot(pilot, "Aggressive fallback pilot")]
    elif mode == 'learning':
        # Show up to 3 directional candidates as pilots
        dir_list = []
        for c in finals_pool:
            n_ok = c.get('n', 0) >= 0.5 * (c.get('min_n', 0) or 0)
            p_ok = (c.get('p') is not None) and (not np.isnan(c.get('p'))) and (c.get('p') < 0.25)
            eff_ok = abs(c.get('effect_abs', 0.0)) >= 0.5 * (c.get('effect_floor', 0.0) or 0.0)
            fin_ok = (c.get('expected_$', 0.0) or 0.0) >= 0.5 * float(cfg.get('FINANCIAL_FLOOR', 0.0) or 0.0)
            if n_ok or p_ok or eff_ok or fin_ok:
                dir_list.append(c)
        dir_list = sorted(dir_list, key=lambda x: x.get('score', 0), reverse=True)[:3]
        out["pilot_actions"] = [_mk_pilot(d, "Learning mode: experimental candidate") for d in dir_list]
        # If none found, provide a fallback pilot regardless of Actions presence
        if not out["pilot_actions"] and len(final) > 0:
            pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
            out["pilot_actions"] = [_mk_pilot(pilot, "Learning fallback pilot")]

    out["confidence_mode"] = mode

    # Relaxed pilot fallback (concierge MVP): propose Pilots when min_n passed,
    # but only for plays that do NOT already have an Action; also dedupe by (play_id, variant_id)
    if not out.get("pilot_actions"):
        actions_pairs = {(str(a.get('play_id')), str(a.get('variant_id', 'base'))) for a in out.get('actions', [])}
        actions_plays = {str(a.get('play_id')) for a in out.get('actions', [])}
        backlog_pairs = {(str(b.get('play_id')), str(b.get('variant_id', 'base'))) for b in out.get('backlog', [])}

        eligible_min_n = []
        for c in finals_pool:
            passed = set(c.get('passed', []))
            has_basic = (c.get('n', 0) or 0) > 0 and bool(c.get('metric'))
            pid = str(c.get('play_id'))
            vid = str(c.get('variant_id', 'base'))
            if not has_basic:
                continue
            if 'min_n' not in passed:
                continue
            # Skip plays already selected as Actions, and skip exact pairs already in Actions/Backlog
            if pid in actions_plays:
                continue
            if (pid, vid) in actions_pairs or (pid, vid) in backlog_pairs:
                continue
            eligible_min_n.append(c)

        if eligible_min_n:
            eligible_min_n = sorted(eligible_min_n, key=lambda x: x.get('score', 0), reverse=True)[:2]
            pilots = [_mk_pilot(c, "Concierge MVP: min_n met; significance shown as confidence badge") for c in eligible_min_n]
            # Final self-dedupe across pilots by (play_id, variant_id)
            seen: set[tuple[str, str]] = set()
            deduped = []
            for p in pilots:
                key = (str(p.get('play_id')), str(p.get('variant_id', 'base')))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(p)
            out["pilot_actions"] = deduped

    # Global dedupe rule: remove any pilots that duplicate Actions/Backlog by play or exact variant pair
    if out.get("pilot_actions"):
        actions_pairs = {(str(a.get('play_id')), str(a.get('variant_id', 'base'))) for a in out.get('actions', [])}
        actions_plays = {str(a.get('play_id')) for a in out.get('actions', [])}
        backlog_pairs = {(str(b.get('play_id')), str(b.get('variant_id', 'base'))) for b in out.get('backlog', [])}
        new_pilots = []
        seen_pairs: set[tuple[str, str]] = set()
        for p in out.get('pilot_actions', []):
            pid = str(p.get('play_id'))
            vid = str(p.get('variant_id', 'base'))
            pair = (pid, vid)
            if pid in actions_plays:
                continue
            if pair in actions_pairs or pair in backlog_pairs:
                continue
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            new_pilots.append(p)
        # Collapse to at most one variant per play: keep highest-score variant
        best_by_play: dict[str, dict] = {}
        for p in new_pilots:
            pid = str(p.get('play_id'))
            if pid not in best_by_play:
                best_by_play[pid] = p
            else:
                if float(p.get('score', 0) or 0) > float(best_by_play[pid].get('score', 0) or 0):
                    best_by_play[pid] = p
        out['pilot_actions'] = list(best_by_play.values())

    # Audience overlap adjustment for Top Actions (conservative)
    try:
        segments_dir = Path(receipts_dir).parent / "segments"
        seen_customers: set[str] = set()

        def _load_segment_customers(attachment: str) -> set[str]:
            if not attachment:
                return set()
            p = segments_dir / attachment
            if not p.exists():
                return set()
            try:
                df = pd.read_csv(p)
                col = None
                for c in df.columns:
                    if str(c).lower() in {"customer_id", "customer", "email", "id"}:
                        col = c; break
                if col is None:
                    return set()
                return set(df[col].astype(str).tolist())
            except Exception:
                return set()

        for idx, a in enumerate(out["actions"]):
            att = a.get("attachment")
            aud = _load_segment_customers(att)
            total = len(aud)
            if total == 0:
                continue
            overlap_n = len(aud & seen_customers)
            overlap_ratio = (overlap_n / total) if total > 0 else 0.0
            if overlap_ratio > 0:
                exp0 = float(a.get("expected_$") or 0.0)
                exp1 = exp0 * max(0.0, 1.0 - overlap_ratio)
                a["expected_$"] = round(exp1, 2)
                a["audience_size_effective"] = total - overlap_n
                a["overlap_with_prior"] = round(overlap_ratio, 3)
            seen_customers |= aud
    except Exception:
        pass

    # Campaign interaction effects (pairwise dampening, env-configurable)
    try:
        interaction_factors = get_interaction_factors(cfg or {})
        prior_play_ids: list[str] = []
        for a in out["actions"]:
            pid = str(a.get("play_id") or "")
            factor = 1.0
            notes: list[str] = []
            for prior in prior_play_ids:
                f = interaction_factors.get((prior, pid))
                if f is not None and f < 1.0:
                    factor *= float(f)
                    notes.append(f"{prior}→{pid} x{f:.2f}")
            if factor < 1.0:
                exp0 = float(a.get("expected_$") or 0.0)
                a["expected_$"] = round(exp0 * factor, 2)
                a["interaction_factor"] = round(factor, 3)
                a["interaction_notes"] = notes
            prior_play_ids.append(pid)
    except Exception:
        pass

    # Optional: log selections for cooldown tracking
    try:
        write_actions_log(receipts_dir, out.get("actions", []))
    except Exception:
        pass

    # Phase 0: emit candidate_debug.json for observability (no behavior change)
    try:
        debug = {
            "window_days": int(aligned.get("window_days") or 28),
            "confidence_mode": mode,
            "counts": {
                "base_candidates": int(len(final)),
                "variants": int(len(finals_for_selection)),
                "actions": int(len(out.get("actions", []))),
                "pilots": int(len(out.get("pilot_actions", []))),
                "watchlist": int(len(out.get("watchlist", []))),
            },
            "candidates": [
                {
                    k: c.get(k)
                    for k in (
                        "id","play_id","metric","n","effect_abs","p","q","ci_low","ci_high",
                        "expected_$","min_n","effect_floor","audience_size","passed","failed","score","tier","reasons"
                    )
                }
                for c in final
            ],
            "actions": [
                {
                    k: a.get(k)
                    for k in (
                        "id","play_id","variant_id","metric","expected_$","score","notes","confidence_label"
                    )
                }
                for a in out.get("actions", [])
            ],
            "pilot_actions": [
                {
                    k: p.get(k)
                    for k in (
                        "id","play_id","variant_id","metric","expected_$","score","notes","confidence_label","pilot_audience_fraction","pilot_budget_cap"
                    )
                }
                for p in out.get("pilot_actions", [])
            ],
        }
        from .utils import write_json as _wj
        _wj(str(Path(receipts_dir) / "candidate_debug.json"), debug)
    except Exception:
        pass

    return out

# Add new function for post-implementation tracking
def track_implementation_status(
    receipts_dir: str,
    action_id: str,
    implemented: bool,
    notes: str = None,
    channels: List[str] = None,
    audience_size: int = None
) -> Dict[str, Any]:
    """
    Call this after implementing (or skipping) an action.
    Can be called from CLI or integrated into your workflow.
    """
    tracker = ActionTracker(receipts_dir)
    return tracker.update_implementation(
        action_id=action_id,
        implemented=implemented,
        implementation_notes=notes,
        channels_used=channels,
        audience_size_actual=audience_size
    )

def track_action_results(
    receipts_dir: str,
    action_id: str,
    revenue: float,
    orders: int = None,
    conversion_rate: float = None,
    notes: str = None
) -> Dict[str, Any]:
    """
    Call this after measuring results (typically 14 days later).
    """
    tracker = ActionTracker(receipts_dir)
    return tracker.track_results(
        action_id=action_id,
        actual_revenue=revenue,
        actual_orders=orders,
        actual_conversion_rate=conversion_rate,
        # Monthly plan: default measurement period ≈ 28 days
        measurement_period_days=28,
        notes=notes
    )

def get_weekly_performance_report(receipts_dir: str) -> str:
    """Generate performance report for the weekly briefing."""
    tracker = ActionTracker(receipts_dir)
    return tracker.generate_weekly_performance_report()

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

    # 2) Rank pool by score (desc) with LTV-aware tiebreaker (non-gating)
    pool_sorted = sorted(
        pool,
        key=lambda x: (x.get("score", 0), x.get("audience_ltv90", 0.0)),
        reverse=True
    )

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
    # Helper: recent window and weekly scalars
    maxd_all = pd.to_datetime(g["Created at"]).max()
    win_days = int(aligned.get("window_days") or 28)
    recent_start_win = maxd_all - pd.Timedelta(days=win_days - 1)
    weeks_in_window = max(1.0, win_days / 7.0)
    # Weekly baseline from recent window net_sales to cap unrealistic lifts
    try:
        grw = g[(g["Created at"] >= recent_start_win)].copy()
        weekly_baseline = float(grw.get("net_sales", pd.Series(dtype=float)).sum()) / weeks_in_window
    except Exception:
        weekly_baseline = 0.0

    # Repeat rate (in-window definition: share of customers with 2+ orders within the window)
    # Build recent/prior windows aligned with win_days
    recent_end = maxd_all
    recent_start = recent_end - pd.Timedelta(days=win_days - 1)
    prior_end = recent_start - pd.Timedelta(seconds=1)
    prior_start = prior_end - pd.Timedelta(days=win_days - 1)

    def _repeat_share_in_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        w = df[(df["Created at"] >= start) & (df["Created at"] <= end)].copy()
        if w.empty:
            return 0, 0, 0.0
        per = (w.groupby("customer_id")["Name"].nunique() if 'Name' in w.columns else w.groupby("customer_id").size())
        x = int((per > 1).sum())
        n = int(per.shape[0])
        rate = float(x / n) if n > 0 else 0.0
        return x, n, rate

    x1, n1, rate_recent = _repeat_share_in_window(g, recent_start, recent_end)
    x2, n2, rate_prior  = _repeat_share_in_window(g, prior_start, prior_end)

    pval = two_proportion_z_test(x1, n1, x2, n2) if (n1 and n2) else 1.0
    # Wilson CI for each period; derive conservative CI for difference
    try:
        from .stats import wilson_ci
        r1_lo, r1_hi = wilson_ci(x1, n1, alpha=0.05) if n1 else (0.0, 0.0)
        r0_lo, r0_hi = wilson_ci(x2, n2, alpha=0.05) if n2 else (0.0, 0.0)
        ci_lo_diff = r1_lo - r0_hi
        ci_hi_diff = r1_hi - r0_lo
    except Exception:
        ci_lo_diff = None; ci_hi_diff = None
    effect_pts  = rate_recent - rate_prior  # absolute delta in points (e.g., +0.024)

    # Heuristic expected value: convert repeat rate delta into revenue proxy.
    # Use recent customer count (n1) and prior repeat as a baseline scaler.
    prior_repeat_baseline = rate_prior if rate_prior > 0 else 0.15
    expected = max(0.0, effect_pts) * n1 * prior_repeat_baseline * gross_margin

    cands.append({
        "id": "repeat_rate_improve",
        "play_id": "winback_21_45",
        "metric": "repeat_rate",
        "n": n1 + n2,
        "effect_abs": effect_pts,
        "p": pval,
        "q": np.nan,                     # set later by BH
        "ci_low": ci_lo_diff, "ci_high": ci_hi_diff,
        "expected_$": expected,
        "min_n": cfg["MIN_N_WINBACK"],
        "effect_floor": cfg["REPEAT_PTS_FLOOR"],
        "rationale": f"Repeat share {rate_recent:.1%} vs {rate_prior:.1%} (Δ {effect_pts:+.1%}).",
        "audience_size": n1,
        "attachment": "segment_winback_21_45.csv",
        "baseline_rate": prior_repeat_baseline,
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
        try:
            from .stats import wilson_ci
            d1_lo, d1_hi = wilson_ci(x1, n1, alpha=0.05)
            d0_lo, d0_hi = wilson_ci(x2, n2, alpha=0.05)
            ci2_lo, ci2_hi = d1_lo - d0_hi, d1_hi - d0_lo
        except Exception:
            ci2_lo = None; ci2_hi = None
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
            "ci_low": ci2_lo, "ci_high": ci2_hi,
            "expected_$": expected2,
            "min_n": cfg["MIN_N_SKU"],
            "effect_floor": cfg["DISCOUNT_PTS_FLOOR"],
            "rationale": f"Discount share {x1/(n1 or 1):.1%} vs {x2/(n2 or 1):.1%} (Δ {effect_pts2:+.1%} reduction).",
            "audience_size": n1,
            "attachment": "segment_discount_hygiene.csv",
            "baseline_rate": aligned["prior_repeat_rate"] or 0.15,
        })

    # --- Subscription nudge: customers with ≥3 orders of the same product in 90 days ---
    try:
        maxd2 = maxd_all
        start90 = maxd2 - pd.Timedelta(days=90)
        gg = g[g["Created at"] >= start90].copy()
        if ("lineitem_any" in gg.columns) or ("products_concat" in gg.columns) or ("Lineitem name" in gg.columns):
            rep = build_g_items(gg)
            # Choose product column: prefer base when normalization is enabled
            prod_col = 'product_key'
            try:
                if bool(cfg.get('FEATURES_PRODUCT_NORMALIZATION', False)) and 'product_key_base' in rep.columns:
                    prod_col = 'product_key_base'
            except Exception:
                pass
            # Per-product threshold using vertical + product detection
            rep["_thr"] = rep[prod_col].astype(str).apply(lambda s: subscription_threshold_for_product(s, cfg))
            cohort = rep[rep['orders_product'] >= rep["_thr"]]
            audience = int(cohort['customer_id'].nunique())
            aov_recent = float(aligned.get("recent_aov") or aligned.get("L28_aov") or np.nan)
            if np.isnan(aov_recent):
                aov_recent = float(np.nanmean(g.get("AOV", []))) if "AOV" in g.columns else 0.0
            # Weekly orders by audience in recent window
            aud_ids = set(cohort['customer_id'].astype(str))
            recent_orders_aud = g[(g["Created at"] >= recent_start_win) & (g["customer_id"].astype(str).isin(aud_ids))]
            weekly_orders = float(recent_orders_aud['Name'].nunique() if 'Name' in recent_orders_aud.columns else len(recent_orders_aud)) / weeks_in_window
            # Conservative weekly uplift rate for subscription (spread over ~12 weeks)
            sub_weekly_uplift = 0.25 / 12.0
            expected = max(0.0, weekly_orders * (aov_recent or 0.0) * sub_weekly_uplift * gross_margin)
            # Compliance-aware adjustment for supplements: dampen expected if observed intervals imply poor compliance
            try:
                # products in cohort
                products = cohort[prod_col].astype(str).unique().tolist()
                comp_factors = []
                for p_name in products:
                    ptype, supply_days = categorize_product(p_name)
                    if ptype != 'supplement':
                        continue
                    orders_p = gg[gg['lineitem_any'].astype(str) == p_name].copy()
                    orders_p = orders_p.sort_values('Created at')
                    med_ipi = orders_p['Created at'].diff().dt.days.median()
                    if pd.isna(med_ipi):
                        continue
                    if med_ipi > supply_days * 1.5:
                        comp_factors.append(0.5)
                    elif med_ipi > supply_days * 1.2:
                        comp_factors.append(0.75)
                    else:
                        comp_factors.append(1.0)
                if comp_factors:
                    expected *= float(np.mean(comp_factors))
            except Exception:
                pass
            # Cap at 25% of weekly baseline to avoid spikes
            if weekly_baseline > 0:
                expected = min(expected, 0.25 * weekly_baseline)
            # Empirical baseline + power check for conversion in next 28d
            p_sub = np.nan
            baseline_conv = None
            try:
                A_start = maxd2 - pd.Timedelta(days=180)
                A_end   = maxd2 - pd.Timedelta(days=90)
                B_start = maxd2 - pd.Timedelta(days=270)
                B_end   = maxd2 - pd.Timedelta(days=180)
                def _conv_next28(start, end):
                    w = g[(g["Created at"] >= start) & (g["Created at"] <= end)].copy()
                    if w.empty or "lineitem_any" not in w.columns:
                        return (0, 0)
                    r = (w.groupby(["customer_id", "lineitem_any"])['Name']
                           .nunique().reset_index(name='orders_product'))
                    aud_ids_local = set(r[r['orders_product'] >= 3]['customer_id'].astype(str))
                    if not aud_ids_local:
                        return (0, 0)
                    after = g[(g["Created at"] > end) & (g["Created at"] <= end + pd.Timedelta(days=28))]
                    conv_set = set(after['customer_id'].astype(str)) if not after.empty else set()
                    nloc = len(aud_ids_local)
                    xloc = sum(1 for c in aud_ids_local if c in conv_set)
                    return (xloc, nloc)
                xA, nA = _conv_next28(A_start, A_end)
                xB, nB = _conv_next28(B_start, B_end)
                if (nA>0 and nB>0):
                    baseline_conv = (xA + xB) / max(1, (nA + nB))
                    p_sub = two_proportion_z_test(xA, nA, xB, nB)
            except Exception:
                baseline_conv = None

            # Power requirement: ensure audience meets minimum to detect an absolute +5pt lift
            expected_delta = 0.05
            baseline_for_power = float(baseline_conv) if baseline_conv is not None else 0.15
            try:
                n_needed = int(required_n_for_proportion(baseline_for_power, expected_delta, alpha=0.05, power=0.8))
            except Exception:
                n_needed = int(cfg.get("MIN_N_SKU", 60))

            if audience >= max(50, int(cfg.get("MIN_N_SKU", 60) // 2)):
                cands.append({
                    "id": "subscription_nudge",
                    "play_id": "subscription_nudge",
                    "metric": "subscription",
                    "n": audience,
                    "effect_abs": 0.05,   # weekly proxy effect (selection heuristic)
                    "p": p_sub,           # empirical if available; otherwise NaN
                    "q": np.nan,
                    "ci_low": None, "ci_high": None,
                    "expected_$": expected,
                    # Use power-based minimum N if higher than config minimum
                    "min_n": max(int(cfg.get("MIN_N_SKU", 60)), n_needed),
                    "effect_floor": 0.05,
                    "rationale": f"Found {audience} customers with ≥3 purchases of the same product in 90d — ideal for subscription. Power check: need ≈ {n_needed} to detect +5 pts.",
                    "audience_size": audience,
                    "attachment": "segment_subscription_nudge.csv",
                    "baseline_rate": baseline_for_power,
                })
    except Exception:
        pass

    # --- Sample to full-size: buyers of sample/travel 14–21 days ago with no subsequent full-size ---
    try:
        maxd3 = pd.to_datetime(g["Created at"]).max()
        win_start = maxd3 - pd.Timedelta(days=21)
        win_end   = maxd3 - pd.Timedelta(days=14)
        gg2 = g[(g["Created at"] >= win_start) & (g["Created at"] <= win_end)].copy()
        if "lineitem_any" in gg2.columns:
            # Identify sample/travel orders by simple token match
            li = gg2["lineitem_any"].astype(str).str.lower()
            sample_mask = li.str.contains(r"sample|travel|mini|trial", regex=True)
            sample_orders = gg2[sample_mask]
            if not sample_orders.empty:
                sample_custs = set(sample_orders["customer_id"].astype(str))
                # Check if they bought a likely full-size later (after win_end until anchor)
                after = g[g["Created at"] > win_end].copy()
                is_full = (~after["lineitem_any"].astype(str).str.lower().str.contains(r"sample|travel|mini|trial", regex=True))
                full_buys = set(after[is_full]["customer_id"].astype(str))
                targets = sample_custs.difference(full_buys)
                audience2 = int(len(targets))
                if audience2 > 0:
                    aov_recent = float(aligned.get("recent_aov") or aligned.get("L28_aov") or np.nan)
                    if np.isnan(aov_recent):
                        aov_recent = float(np.nanmean(g.get("AOV", []))) if "AOV" in g.columns else 0.0
                    # Weekly expected conversions: approx 35% over ~3 weeks => ~11.7% weekly
                    weekly_sample_orders = float(gg2.shape[0]) / weeks_in_window
                    conv_weekly = 0.35 / 3.0
                    expected_sf = max(0.0, weekly_sample_orders * conv_weekly * (aov_recent or 0.0) * gross_margin)
                    if weekly_baseline > 0:
                        expected_sf = min(expected_sf, 0.25 * weekly_baseline)
                    # Effect heuristic and empirical p-value using two time cohorts as proxy
                    effect_sf = 0.12
                    # Empirical p-value: compare conversion within 21d for sample buyers in two prior windows
                    try:
                        # Cohort A: 35–56d before anchor (treatment-like)
                        A_start = maxd3 - pd.Timedelta(days=56)
                        A_end   = maxd3 - pd.Timedelta(days=35)
                        # Cohort B: 70–91d before anchor (baseline-like)
                        B_start = maxd3 - pd.Timedelta(days=91)
                        B_end   = maxd3 - pd.Timedelta(days=70)
                        def conv_rate(start, end):
                            w = g[(g["Created at"] >= start) & (g["Created at"] <= end)].copy()
                            if w.empty: return (0, 0)
                            liw = w["lineitem_any"].astype(str).str.lower()
                            samp_m = liw.str.contains(r"sample|travel|mini|trial", regex=True)
                            samp_orders = w[samp_m].copy()
                            if samp_orders.empty: return (0, 0)
                            # next 21 days window for conversion to full-size
                            next_horizon_end = end + pd.Timedelta(days=21)
                            after = g[(g["Created at"] > end) & (g["Created at"] <= next_horizon_end)].copy()
                            is_full2 = (~after["lineitem_any"].astype(str).str.lower().str.contains(r"sample|travel|mini|trial", regex=True))
                            full_set = set(after[is_full2]["customer_id"].astype(str))
                            custs = samp_orders["customer_id"].astype(str).unique().tolist()
                            x = sum(1 for c in custs if c in full_set)
                            n = len(custs)
                            return (x, n)
                        xA, nA = conv_rate(A_start, A_end)
                        xB, nB = conv_rate(B_start, B_end)
                        if nA > 0 and nB > 0:
                            p_sf = two_proportion_z_test(xA, nA, xB, nB)
                            try:
                                from .stats import wilson_ci
                                sA_lo, sA_hi = wilson_ci(xA, nA, alpha=0.05)
                                sB_lo, sB_hi = wilson_ci(xB, nB, alpha=0.05)
                                ci_sf_lo, ci_sf_hi = sA_lo - sB_hi, sA_hi - sB_lo
                            except Exception:
                                ci_sf_lo = None; ci_sf_hi = None
                        else:
                            p_sf = 0.06 if audience2 < 40 else 0.02
                            ci_sf_lo = None; ci_sf_hi = None
                    except Exception:
                        p_sf = 0.06 if audience2 < 40 else 0.02
                    cands.append({
                        "id": "sample_to_full",
                        "play_id": "sample_to_full",
                        "metric": "sample_to_full",
                        "n": audience2,
                        "effect_abs": effect_sf,
                        "p": p_sf,
                        "q": np.nan,
                        "ci_low": ci_sf_lo, "ci_high": ci_sf_hi,
                        "expected_$": expected_sf,
                        "min_n": int(cfg.get("MIN_N_SKU", 60)),
                        "effect_floor": 0.05,
                        "rationale": f"{audience2} recent sample/travel buyers (14–21d) without full-size — prime for follow-up.",
                        "audience_size": audience2,
                        "attachment": "segment_sample_to_full.csv",
                        "baseline_rate": None,
                    })
    except Exception:
        pass

    # --- Routine builder: skincare single-product purchasers (bundle opportunity) ---
    try:
        anchor = pd.to_datetime(g["Created at"]).max()
        recent_start = anchor - pd.Timedelta(days=60)
        lookback_start = anchor - pd.Timedelta(days=90)
        gr = g[(g["Created at"] >= recent_start)].copy()
        # Focus on skincare category
        if "category" in gr.columns:
            gr_skin = gr[gr["category"].astype(str).str.lower() == "skincare"].copy()
        else:
            gr_skin = gr.copy()
        # Candidates: customers in skincare recently
        cand_ids = set(gr_skin["customer_id"].astype(str))
        if cand_ids:
            gl = g[(g["Created at"] >= lookback_start)].copy()
            gl["customer_id"] = gl["customer_id"].astype(str)
            # Distinct products in lookback per customer (prefer g_items)
            try:
                gi2 = build_g_items(gl)
                if gi2 is not None and not gi2.empty:
                    k = gi2.groupby("customer_id")["product_key"].nunique()
                    single_prod_ids = set(k[k <= 1].index)
                else:
                    raise ValueError('g_items empty')
            except Exception:
                if "lineitem_any" in gl.columns:
                    k = gl.groupby("customer_id")["lineitem_any"].nunique()
                    single_prod_ids = set(k[k <= 1].index)
                else:
                    single_prod_ids = set()
            targets = list(cand_ids.intersection(single_prod_ids))
            audience_rb = int(len(targets))
            if audience_rb > 0:
                aov_recent = float(aligned.get("recent_aov") or aligned.get("L28_aov") or np.nan)
                if np.isnan(aov_recent):
                    aov_recent = float(np.nanmean(g.get("AOV", []))) if "AOV" in g.columns else 0.0
                # Weekly orders by audience in recent window
                rb_ids = set(targets)
                recent_orders_rb = g[(g["Created at"] >= recent_start_win) & (g["customer_id"].astype(str).isin(rb_ids))]
                weekly_orders_rb = float(recent_orders_rb['Name'].nunique() if 'Name' in recent_orders_rb.columns else len(recent_orders_rb)) / weeks_in_window
                expected_rb = max(0.0, weekly_orders_rb * (aov_recent or 0.0) * 0.40 * gross_margin)  # 40% AOV lift proxy
                if weekly_baseline > 0:
                    expected_rb = min(expected_rb, 0.25 * weekly_baseline)
                effect_rb = 0.08  # weekly proxy effect
                # Empirical p-value: Welch t-test on AOV for audience across two prior 60d cohorts
                try:
                    def _aov_for_ids(start, end, ids):
                        ww = g[(g["Created at"] >= start) & (g["Created at"] <= end)].copy()
                        if ww.empty or "AOV" not in ww.columns:
                            return np.array([])
                        ww = ww[ww["customer_id"].astype(str).isin(ids)]
                        return ww["AOV"].astype(float).values
                    A_start = anchor - pd.Timedelta(days=120)
                    A_end   = anchor - pd.Timedelta(days=60)
                    B_start = anchor - pd.Timedelta(days=180)
                    B_end   = anchor - pd.Timedelta(days=120)
                    aA = _aov_for_ids(A_start, A_end, rb_ids)
                    aB = _aov_for_ids(B_start, B_end, rb_ids)
                    p_rb = welch_t_test(aA, aB) if (aA.size>1 and aB.size>1) else (0.06 if audience_rb<80 else 0.03)
                except Exception:
                    p_rb = 0.06 if audience_rb<80 else 0.03
                cands.append({
                    "id": "routine_builder",
                    "play_id": "routine_builder",
                    "metric": "bundle_aov",
                    "n": audience_rb,
                    "effect_abs": effect_rb,
                    "p": p_rb,
                    "q": np.nan,
                    "ci_low": None, "ci_high": None,
                    "expected_$": expected_rb,
                    "min_n": int(cfg.get("MIN_N_SKU", 60)),
                    "effect_floor": float(cfg.get("AOV_EFFECT_FLOOR", 0.03)),
                    "rationale": f"{audience_rb} skincare single-product buyers — bundle to complete routine.",
                    "audience_size": audience_rb,
                    "attachment": "segment_routine_builder.csv",
                    "baseline_rate": None,
                })
    except Exception:
        pass

    # --- Ingredient education: first-time buyers of technical ingredients ---
    try:
        tech_re = r"retinol|acid|aha|bha|salicy|glycolic|lactic|peptide|niacinamide|vitamin c|ascorb"
        # First-time overall buyers who purchased in recent window and item matches technical keyword
        freq_all = g.groupby("customer_id")["Name"].nunique().rename("orders_total")
        recent = g[g["Created at"] >= recent_start_win].copy().merge(freq_all, left_on="customer_id", right_index=True, how="left")
        if "lineitem_any" in recent.columns:
            mask = recent["orders_total"].fillna(0).astype(int).eq(1) & recent["lineitem_any"].astype(str).str.lower().str.contains(tech_re, regex=True)
            edu_targets = recent.loc[mask, "customer_id"].astype(str).unique().tolist()
            audience_edu = int(len(edu_targets))
            aov_recent = float(aligned.get("recent_aov") or aligned.get("L28_aov") or np.nan)
            if np.isnan(aov_recent):
                aov_recent = float(np.nanmean(g.get("AOV", []))) if "AOV" in g.columns else 0.0
            # Weekly orders among this audience in recent window
            edu_ids = set(edu_targets)
            recent_orders_edu = g[(g["Created at"] >= recent_start_win) & (g["customer_id"].astype(str).isin(edu_ids))]
            weekly_orders_edu = float(recent_orders_edu['Name'].nunique() if 'Name' in recent_orders_edu.columns else len(recent_orders_edu)) / weeks_in_window
            # Education improves retention modestly; assume 25% over ~8 weeks => ~3.1% weekly
            edu_weekly_uplift = 0.25 / 8.0
            expected_edu = max(0.0, weekly_orders_edu * (aov_recent or 0.0) * edu_weekly_uplift * gross_margin)
            if weekly_baseline > 0:
                expected_edu = min(expected_edu, 0.25 * weekly_baseline)
            # Empirical p-value: repeat within 60 days for two historic cohorts
            try:
                def _repeat_within_60(start, end):
                    w = g[(g["Created at"] >= start) & (g["Created at"] <= end)].copy()
                    if w.empty or "lineitem_any" not in w.columns:
                        return (0, 0)
                    freq = w.groupby("customer_id")["Name"].nunique().rename("orders_total")
                    w = w.merge(freq, left_on="customer_id", right_index=True, how="left")
                    m = w["orders_total"].fillna(0).astype(int).eq(1) & w["lineitem_any"].astype(str).str.lower().str.contains(tech_re, regex=True)
                    custs = set(w.loc[m, "customer_id"].astype(str))
                    if not custs:
                        return (0, 0)
                    after = g[(g["Created at"] > end) & (g["Created at"] <= end + pd.Timedelta(days=60))]
                    conv = set(after["customer_id"].astype(str)) if not after.empty else set()
                    return (sum(1 for c in custs if c in conv), len(custs))
                A_start = maxd_all - pd.Timedelta(days=120)
                A_end   = maxd_all - pd.Timedelta(days=60)
                B_start = maxd_all - pd.Timedelta(days=180)
                B_end   = maxd_all - pd.Timedelta(days=120)
                xA, nA = _repeat_within_60(A_start, A_end)
                xB, nB = _repeat_within_60(B_start, B_end)
                p_edu = two_proportion_z_test(xA, nA, xB, nB) if (nA>0 and nB>0) else (0.06 if audience_edu<100 else 0.04)
            except Exception:
                p_edu = 0.06 if audience_edu<100 else 0.04
            if audience_edu >= max(40, int(cfg.get("MIN_N_SKU", 60) // 2)):
                cands.append({
                    "id": "ingredient_education",
                    "play_id": "ingredient_education",
                    "metric": "retention",
                    "n": audience_edu,
                    "effect_abs": 0.03,
                    "p": p_edu,
                    "q": np.nan,
                    "ci_low": None, "ci_high": None,
                    "expected_$": expected_edu,
                    "min_n": int(cfg.get("MIN_N_SKU", 60)),
                    "effect_floor": 0.03,
                    "rationale": f"{audience_edu} first-time technical buyers identified — education boosts retention.",
                    "audience_size": audience_edu,
                    "attachment": "segment_ingredient_education.csv",
                    "baseline_rate": None,
                })
    except Exception:
        pass

    # --- Empty bottle reminder: size-based depletion window ---
    try:
        # Last purchase per customer
        last = g.sort_values("Created at").groupby("customer_id").tail(1).copy()
        if "lineitem_any" in last.columns and "days_since_last" in last.columns:
            names = last["lineitem_any"].astype(str).str.lower()
            dsl = last["days_since_last"].astype(float)
            # crude size parsing
            size_days = []
            for s in names:
                if "100ml" in s or "3.4 oz" in s or "3.4oz" in s:
                    size_days.append(75)
                elif "50ml" in s or "1.7 oz" in s or "1.7oz" in s:
                    size_days.append(40)
                elif "30ml" in s or "1 oz" in s or "1oz" in s:
                    size_days.append(25)
                else:
                    size_days.append(None)
            last = last.assign(_deplete_days=size_days)
            window = last[~pd.isna(last["_deplete_days"])].copy()
            # Target if within +/- 3 days of depletion
            m = (window["days_since_last"] >= (window["_deplete_days"] - 3)) & (window["days_since_last"] <= (window["_deplete_days"] + 3))
            targets_eb = window.loc[m, "customer_id"].astype(str).unique().tolist()
            audience_eb = int(len(targets_eb))
            if audience_eb > 0:
                aov_recent = float(aligned.get("recent_aov") or aligned.get("L28_aov") or np.nan)
                if np.isnan(aov_recent):
                    aov_recent = float(np.nanmean(g.get("AOV", []))) if "AOV" in g.columns else 0.0
                # Assume ~10% weekly conversion on timely reminders
                weekly_reminders = audience_eb  # approximate per week at steady-state
                conv_weekly = 0.10
                expected_eb = max(0.0, weekly_reminders * conv_weekly * (aov_recent or 0.0) * gross_margin)
                if weekly_baseline > 0:
                    expected_eb = min(expected_eb, 0.25 * weekly_baseline)
                # Empirical p-value: near-depletion reorder within 14 days for two historic cohorts
                try:
                    def _deplete_conv(start, end):
                        ww = g[(g["Created at"] >= start) & (g["Created at"] <= end)].copy()
                        if ww.empty or "lineitem_any" not in ww.columns or "days_since_last" not in ww.columns:
                            return (0, 0)
                        nm = ww["lineitem_any"].astype(str).str.lower()
                        size_days = np.where(nm.str.contains("100ml|3.4 oz|3.4oz"), 75,
                                     np.where(nm.str.contains("50ml|1.7 oz|1.7oz"), 40,
                                     np.where(nm.str.contains("30ml|1 oz|1oz"), 25, np.nan)))
                        ww = ww.assign(_deplete_days=size_days)
                        w2 = ww[~pd.isna(ww["_deplete_days"])].copy()
                        if w2.empty:
                            return (0, 0)
                        m2 = (w2["days_since_last"] >= (w2["_deplete_days"] - 3)) & (w2["days_since_last"] <= (w2["_deplete_days"] + 3))
                        custs = set(w2.loc[m2, "customer_id"].astype(str))
                        if not custs:
                            return (0, 0)
                        after = g[(g["Created at"] > end) & (g["Created at"] <= end + pd.Timedelta(days=14))]
                        conv = set(after["customer_id"].astype(str)) if not after.empty else set()
                        return (sum(1 for c in custs if c in conv), len(custs))
                    A_start = maxd_all - pd.Timedelta(days=120)
                    A_end   = maxd_all - pd.Timedelta(days=60)
                    B_start = maxd_all - pd.Timedelta(days=180)
                    B_end   = maxd_all - pd.Timedelta(days=120)
                    xA, nA = _deplete_conv(A_start, A_end)
                    xB, nB = _deplete_conv(B_start, B_end)
                    if (nA>0 and nB>0):
                        p_eb = two_proportion_z_test(xA, nA, xB, nB)
                        try:
                            from .stats import wilson_ci
                            eA_lo, eA_hi = wilson_ci(xA, nA, alpha=0.05)
                            eB_lo, eB_hi = wilson_ci(xB, nB, alpha=0.05)
                            ci_eb_lo, ci_eb_hi = eA_lo - eB_hi, eA_hi - eB_lo
                        except Exception:
                            ci_eb_lo = None; ci_eb_hi = None
                    else:
                        p_eb = 0.06 if audience_eb<80 else 0.05
                        ci_eb_lo = None; ci_eb_hi = None
                except Exception:
                    p_eb = 0.06 if audience_eb<80 else 0.05
                    ci_eb_lo = None; ci_eb_hi = None
                cands.append({
                    "id": "empty_bottle",
                    "play_id": "empty_bottle",
                    "metric": "reorder",
                    "n": audience_eb,
                    "effect_abs": conv_weekly,
                    "p": p_eb,
                    "q": np.nan,
                    "ci_low": ci_eb_lo, "ci_high": ci_eb_hi,
                    "expected_$": expected_eb,
                    "min_n": int(cfg.get("MIN_N_SKU", 60)),
                    "effect_floor": 0.03,
                    "rationale": f"{audience_eb} customers near predicted depletion — timely reorder reminder.",
                    "audience_size": audience_eb,
                    "attachment": "segment_empty_bottle.csv",
                    "baseline_rate": None,
                })
    except Exception:
        pass

    return cands

    

def _normalize_aligned(aligned: dict, cfg: dict) -> dict:
    """Accept nested KPI snapshot {L7:{...}, L28:{...}} or flat aligned, return flat structure.
    Chooses window from cfg['CHOSEN_WINDOW'] (default L28).
    """
    if aligned is None:
        return {}
    # If already flat, return as-is
    if 'window_days' in aligned and 'recent_n' in aligned:
        return aligned
    # If nested, flatten according to chosen window
    lbl = 'L7' if str((cfg or {}).get('CHOSEN_WINDOW', 'L28')).upper() == 'L7' else 'L28'
    block = (aligned.get(lbl) or {})
    prior = (block.get('prior') or {})
    days = 7 if lbl == 'L7' else 28
    anchor = aligned.get('anchor')
    # Compute bounds from anchor for completeness
    rs = re = ps = pe = None
    try:
        if anchor is not None:
            anchor_ts = pd.Timestamp(anchor)
            re = anchor_ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
            rs = re.normalize() - pd.Timedelta(days=days - 1)
            pe = rs - pd.Timedelta(seconds=1)
            ps = pe.normalize() - pd.Timedelta(days=days - 1)
    except Exception:
        pass
    return {
        'window_days': days,
        'recent_start': str(rs.date()) if rs is not None else None,
        'recent_end': str(re.date()) if re is not None else None,
        'prior_start': str(ps.date()) if ps is not None else None,
        'prior_end': str(pe.date()) if pe is not None else None,
        'recent_n': int(block.get('orders') or 0),
        'prior_n': int(prior.get('orders') or 0),
        'recent_repeat_rate': float(block.get('repeat_rate') or 0.0) if block.get('repeat_rate') is not None else 0.0,
        'prior_repeat_rate': float(prior.get('repeat_rate') or 0.0) if prior.get('repeat_rate') is not None else 0.0,
        'recent_aov': float(block.get('aov') or 0.0) if block.get('aov') is not None else 0.0,
        'prior_aov': float(prior.get('aov') or 0.0) if prior.get('aov') is not None else 0.0,
        'recent_discount_rate': float(block.get('discount_rate') or 0.0) if block.get('discount_rate') is not None else 0.0,
        'prior_discount_rate': float(prior.get('discount_rate') or 0.0) if prior.get('discount_rate') is not None else 0.0,
        'anchor': anchor,
    }


def select_actions(g, aligned, cfg, playbooks_path: str, receipts_dir: str, policy_path: str | None = None,
                   inventory_metrics: pd.DataFrame | None = None) -> Dict[str, Any]:
    """Public entry: normalize aligned input then delegate to core implementation."""
    aligned_norm = _normalize_aligned(aligned, cfg or {})
    return _select_actions_impl(g, aligned_norm, cfg, playbooks_path, receipts_dir, policy_path, inventory_metrics)

    # LTV signal (non-blocking): compute once
    ltv_info = compute_repeat_curve(g, horizon_days=[60, 90]) if g is not None else {"store": {}, "per_customer": []}
    store_ltv90 = float(((ltv_info or {}).get("store", {}) or {}).get(90, {}).get("ltv", 0.0) or 0.0)
    try:
        arr = np.array([float(x.get("ltv90", 0.0) or 0.0) for x in (ltv_info.get("per_customer", []) or [])], dtype=float)
        ltv90_p90 = float(np.quantile(arr, 0.9)) if arr.size > 0 else 0.0
    except Exception:
        ltv90_p90 = 0.0

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

        # attach LTV estimates for receipts/tiebreakers only (no gating)
        cand["audience_ltv90"] = store_ltv90
        cand["ltv90_top_decile"] = ltv90_p90
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

    # (helper functions already defined earlier in this function; avoid duplicate definitions)
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

            # Base expected value for one exposure window
            vc["expected_$"] = max(0.0, exp_base * lift_mult - incentive_cost)

            # Monthly scaling with fatigue and light seasonality adjustment
            # - Fatigue: diminishing weekly multipliers over 4 weeks
            # - Seasonality: adjust by how L7 compares to avg weekly over L28 (clamped)
            try:
                fatigue_schedule = [1.00, 0.85, 0.70, 0.60]
                monthly_multiplier = float(sum(fatigue_schedule))  # 3.15 instead of 4.0

                l28_ns = float(((aligned or {}).get("L28", {}) or {}).get("net_sales", 0.0) or 0.0)
                l7_ns  = float(((aligned or {}).get("L7",  {}) or {}).get("net_sales", 0.0) or 0.0)
                avg_w  = (l28_ns / 4.0) if l28_ns > 0 else 0.0
                season_ratio = (l7_ns / avg_w) if (avg_w > 0 and l7_ns > 0) else 1.0
                # clamp to avoid over-reacting
                season_factor = max(0.9, min(1.1, season_ratio))

                vc["expected_$"] *= (monthly_multiplier * season_factor)

                # Saturation guardrail: cap single-action monthly lift vs baseline
                monthly_baseline = l28_ns
                if monthly_baseline > 0:
                    cap = 0.35 * monthly_baseline
                    if vc["expected_$"] > cap:
                        vc["expected_$"] = cap
            except Exception:
                # Fallback to simple monthly scaling if inputs are missing
                vc["expected_$"] *= 4.0

            # Non-blocking LTV preference: if audience LTV is top-decile, nudge toward no-discount
            try:
                aud_ltv = float(cand.get("audience_ltv90") or 0.0)
                top_dec = float(cand.get("ltv90_top_decile") or 0.0)
                high_ltv = (aud_ltv >= top_dec) and (top_dec > 0)
            except Exception:
                high_ltv = False
            offer_type = str(v.get("offer_type", "")).lower()
            if high_ltv and ("discount" in offer_type or "percent_of_aov" in str(v.get("cost_type",""))):
                # small penalty on discounted variants for high-LTV audiences
                vc["expected_$"] *= 0.97
            elif high_ltv and (offer_type == "no_discount"):
                vc["expected_$"] *= 1.03

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

            # Inventory-aware adjustments (soft by default)
            if inv_sum is not None:
                play_id = str(vc.get('play_id') or '').lower()
                min_cover = int(float(cover_map.get(play_id, default_cover))) if cover_map else default_cover
                cover_health = inv_sum.get('cover_min', float('inf'))
                trust_mean = float(inv_sum.get('trust_mean', 1.0))
                # Base trust scaling
                vc["expected_$"] *= trust_mean
                vc['inv_trust'] = round(trust_mean, 2)
                # Play-specific logic
                if 'discount_hygiene' in play_id:
                    # Determine scope: product_specific if variant id starts with 'targeted', else sitewide
                    v_id = str(vc.get('variant_id') or '').lower()
                    if v_id.startswith('targeted'):
                        # Product-specific: exclude low-stock SKUs among targeted set
                        skus = _targeted_skus_for_play(vc)
                        if skus:
                            rows = inv_df[inv_df['sku'].astype(str).isin([str(s) for s in skus])].copy()
                            if not rows.empty and 'cover_days' in rows.columns:
                                total = int(len(rows)) or 1
                                low = int((rows['cover_days'] < 14).sum())
                                keep_ratio = max(0.0, min(1.0, (total - low) / total))
                                vc['expected_$'] *= keep_ratio
                                vc.setdefault('notes', []).append(f"Discount scope: product-specific — excluded {low}/{total} low-stock SKUs (≥14d cover required)")
                    else:
                        # Sitewide: apply in-stock ratio across catalog
                        in_stock_ratio = float(inv_sum.get('in_stock_ratio14', 1.0))
                        vc['inv_in_stock_ratio14'] = round(in_stock_ratio, 2)
                        vc["expected_$"] *= in_stock_ratio
                        vc.setdefault('notes', []).append(f"Discount scope: sitewide — in-stock≥14d ratio≈{in_stock_ratio:.0%}")
                # Coverage check
                if cover_health < min_cover:
                    if inv_mode == 'hard':
                        # skip this variant due to low coverage
                        continue
                    # soft: penalize score and annotate
                    vc.setdefault('notes', []).append(f"Inventory cover low (min≈{cover_health:.0f}d < {min_cover}d)")
                    # small penalty
                    vc['score'] = (vc.get('score') or 0.0) * 0.9
                    vc['inv_cover_min'] = float(cover_health)

            # Apply targeted-SKU inventory logic (fulfillment and per-play cover)
            _apply_inventory_to_variant(vc)
            if vc.get('__skip_due_inventory__'):
                # hard skip if coverage too low for targeted SKUs
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

    # Confidence Mode handling
    mode = str((cfg or {}).get("CONFIDENCE_MODE", "conservative")).strip().lower()

    def _mk_pilot(pilot: dict, note: str) -> dict:
        n_needed = pilot.get("min_n", 0)
        if pilot.get("metric") in ("repeat_rate", "discount_rate"):
            p = pilot.get("baseline_rate", 0.15) or 0.15
            delta = pilot.get("effect_floor", 0.02) or 0.02
            n_needed = required_n_for_proportion(p, delta, alpha=0.05, power=0.8)
        exp = float(pilot.get("expected_$", 0.0) or 0.0)
        return {
            **pilot,
            "tier": "Pilot",
            "pilot_audience_fraction": cfg.get("PILOT_AUDIENCE_FRACTION", 0.2),
            "pilot_budget_cap": cfg.get("PILOT_BUDGET_CAP", 200.0),
            "n_needed": int(n_needed),
            "decision_rule": "Graduate to full rollout if CI excludes 0 or q ≤ α at 28 days; else rollback.",
            "confidence_label": pilot.get("confidence_label", "Low"),
            "expected_range": [round(exp * 0.6, 2), round(exp * 1.3, 2)],
            "notes": (pilot.get("notes") or []) + [note],
        }

    finals_pool = finals_for_selection if finals_for_selection else final

    if mode == 'conservative':
        if len(out["actions"]) == 0 and len(final) > 0:
            pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
            out["pilot_actions"] = [_mk_pilot(pilot, "Conservative fallback pilot")]
    elif mode == 'aggressive':
        meds = []
        for c in finals_pool:
            failed = set(c.get('failed', [])); passed = set(c.get('passed', []))
            if ('significance' in failed) and ('min_n' in passed) and ('effect_floor' not in failed) and ('financial_floor' not in failed):
                meds.append(c)
        meds = sorted(meds, key=lambda x: x.get('score', 0), reverse=True)[:2]
        out["pilot_actions"] = [_mk_pilot(m, "Aggressive mode: medium-confidence (fails significance only)") for m in meds]
        if not out["pilot_actions"] and len(out["actions"]) == 0 and len(final) > 0:
            pilot = sorted(final, key=lambda x: x.get("score", 0), reverse=True)[0]
            out["pilot_actions"] = [_mk_pilot(pilot, "Aggressive fallback pilot")]
    elif mode == 'learning':
        dir_list = []
        for c in finals_pool:
            n_ok = c.get('n', 0) >= 0.5 * (c.get('min_n', 0) or 0)
            p_ok = (c.get('p') is not None) and (not np.isnan(c.get('p'))) and (c.get('p') < 0.25)
            eff_ok = abs(c.get('effect_abs', 0.0)) >= 0.5 * (c.get('effect_floor', 0.0) or 0.0)
            fin_ok = (c.get('expected_$', 0.0) or 0.0) >= 0.5 * float(cfg.get('FINANCIAL_FLOOR', 0.0) or 0.0)
            if n_ok or p_ok or eff_ok or fin_ok:
                dir_list.append(c)
        dir_list = sorted(dir_list, key=lambda x: x.get('score', 0), reverse=True)[:3]
        out["pilot_actions"] = [_mk_pilot(d, "Learning mode: experimental candidate") for d in dir_list]

    out["confidence_mode"] = mode

    # --- Audience overlap adjustment for Top Actions (conservative) ---
    # Reduce expected_$ for downstream actions in proportion to their audience overlap
    try:
        segments_dir = Path(receipts_dir).parent / "segments"
        seen_customers: set[str] = set()

        def _load_segment_customers(attachment: str) -> set[str]:
            if not attachment:
                return set()
            p = segments_dir / attachment
            if not p.exists():
                return set()
            try:
                df = pd.read_csv(p)
                col = None
                for c in df.columns:
                    if str(c).lower() in {"customer_id", "customer", "email", "id"}:
                        col = c; break
                if col is None:
                    return set()
                return set(df[col].astype(str).tolist())
            except Exception:
                return set()

        for idx, a in enumerate(out["actions"]):
            att = a.get("attachment")
            aud = _load_segment_customers(att)
            total = len(aud)
            if total == 0:
                continue
            overlap_n = len(aud & seen_customers)
            overlap_ratio = (overlap_n / total) if total > 0 else 0.0
            if overlap_ratio > 0:
                # proportional reduction of expected impact to avoid double counting
                exp0 = float(a.get("expected_$") or 0.0)
                exp1 = exp0 * max(0.0, 1.0 - overlap_ratio)
                a["expected_$"] = round(exp1, 2)
                a["audience_size_effective"] = total - overlap_n
                a["overlap_with_prior"] = round(overlap_ratio, 3)
            # add to seen
            seen_customers |= aud
    except Exception:
        pass

    # --- Campaign interaction effects (pairwise dampening, env-configurable) ---
    try:
        from .utils import get_interaction_factors
        interaction_factors = get_interaction_factors(cfg or {})

        prior_play_ids: list[str] = []
        for a in out["actions"]:
            pid = str(a.get("play_id") or "")
            factor = 1.0
            notes: list[str] = []
            for prior in prior_play_ids:
                f = interaction_factors.get((prior, pid))
                if f is not None and f < 1.0:
                    factor *= float(f)
                    notes.append(f"{prior}→{pid} x{f:.2f}")
            if factor < 1.0:
                exp0 = float(a.get("expected_$") or 0.0)
                a["expected_$"] = round(exp0 * factor, 2)
                a["interaction_factor"] = round(factor, 3)
                a["interaction_notes"] = notes
            prior_play_ids.append(pid)
    except Exception:
        pass

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
    bullets: list[str] = []
    if "discount_hygiene" in pid:
        bullets.append(_receipt_discount_hygiene(aligned, action))
    elif "winback" in pid:
        bullets.append(_receipt_winback(aligned, action))
    elif "bestseller" in pid or "amplify" in pid:
        bullets.append(_receipt_bestseller(aligned, action))
    elif "dormant" in pid:
        bullets.append(_receipt_dormant(aligned, action))
    elif "subscription" in pid:
        aud = action.get("audience_size") or action.get("n")
        est = _money(action.get("expected_$"))
        bullets.append(f"{int(aud or 0)} customers bought the same product ≥3 times in 90d — prime for subscription.")
        bullets.append(f"Expected LTV lift cohort ≈ {est} (heuristic).")
    elif "sample_to_full" in pid:
        aud = action.get("audience_size") or action.get("n")
        est = _money(action.get("expected_$"))
        bullets.append(f"{int(aud or 0)} recent sample/travel buyers (14–21d) without full-size.")
        bullets.append(f"Follow-up offer expected to convert ≈35%; impact ≈ {est}.")
    elif "routine_builder" in pid:
        aud = action.get("audience_size") or action.get("n")
        est = _money(action.get("expected_$"))
        bullets.append(f"{int(aud or 0)} skincare single-product buyers identified in the last 60d.")
        bullets.append(f"Bundle complementary items to lift AOV; expected impact ≈ {est}.")
    else:
        # fallback: use rationale/effect
        eff = action.get("effect_abs")
        bullets.append(action.get("rationale") or f"Effect delta {_pct(eff)} vs prior; expected ≈ {_money(action.get('expected_$'))}.")

    # Append LTV note if available (applies to any play)
    if action.get("audience_ltv90") is not None:
        try:
            ltv = float(action.get("audience_ltv90") or 0.0)
            topd = float(action.get("ltv90_top_decile") or 0.0)
            # Hide if effectively zero to keep receipts crisp
            if ltv >= 1.0:
                decile_note = "top-decile LTV prioritized; no-discount variant" if (topd > 0 and ltv >= topd) else None
                s = f"LTV90 (contrib) ≈ {_money(ltv)}"
                if decile_note:
                    s += f"; {decile_note}"
                bullets.append(s)
        except Exception:
            pass
    return bullets

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
