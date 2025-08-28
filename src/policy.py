from __future__ import annotations
from pathlib import Path
import json

DEFAULT_POLICY = {
    "allow_free_shipping": True,
    "max_discount_pct": 15,
    "channel_caps": {"email_per_week": 2, "sms_per_week": 1},
}

def load_policy(path: str | None = None) -> dict:
    if path and Path(path).exists():
        try: return {**DEFAULT_POLICY, **json.loads(Path(path).read_text())}
        except Exception: pass
    return DEFAULT_POLICY

def is_eligible(expr: str, policy: dict) -> bool:
    expr = (expr or "True").strip()
    if expr.lower() == "true": return True
    if expr.lower() == "false": return False
    try:
        if expr.startswith("policy."):
            tail = expr.split("policy.", 1)[1]
            if any(op in tail for op in (">=", "<=", "==", ">", "<")):
                for tok in (">=", "<=", "==", ">", "<"):
                    if tok in tail:
                        left, right = [x.strip() for x in tail.split(tok)]
                        lval = float(policy.get(left, 0) if isinstance(policy.get(left), (int,float)) else policy.get(left))
                        rval = float(right)
                        return {">=": lval >= rval, "<=": lval <= rval, "==": lval == rval, ">": lval > rval, "<": lval < rval}[tok]
            return bool(policy.get(tail, False))
    except Exception:
        return False
    return False
