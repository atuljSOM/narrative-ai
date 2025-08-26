
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .stats import wilson_ci

def repeat_share_chart(aligned, out_path: str):
    means = [aligned["prior_repeat_rate"], aligned["recent_repeat_rate"]]
    ns = [aligned["prior_n"], aligned["recent_n"]]
    ci = [wilson_ci(int(round((means[0] or 0)* (ns[0] or 1))), ns[0] or 1), wilson_ci(int(round((means[1] or 0)*(ns[1] or 1))), ns[1] or 1)]
    yerr = [[(means[i] or 0)-ci[i][0] for i in range(2)], [ci[i][1]-(means[i] or 0) for i in range(2)]]
    x = np.arange(2); plt.figure(); plt.bar(x, [m or 0 for m in means], yerr=yerr, capsize=4)
    plt.xticks(x, ["Prior","Recent"]); plt.ylabel("Repeat share"); plt.title(f"Repeat share (window {aligned['window_days']}d)"); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close(); return out_path

def aov_chart(aligned, out_path: str):
    means = [aligned["prior_aov"] or 0, aligned["recent_aov"] or 0]; ns = [aligned["prior_n"], aligned["recent_n"]]
    sd = [means[0]*0.5 if means[0] else 0.0, means[1]*0.5 if means[1] else 0.0]
    yerr = [[0 if ns[i]==0 else 1.96*sd[i]/np.sqrt(ns[i]) for i in range(2)], [0 if ns[i]==0 else 1.96*sd[i]/np.sqrt(ns[i]) for i in range(2)]]
    x=np.arange(2); plt.figure(); plt.bar(x, means, yerr=yerr, capsize=4)
    plt.xticks(x, ["Prior","Recent"]); plt.ylabel("AOV"); plt.title(f"AOV (window {aligned['window_days']}d)"); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close(); return out_path

def discount_share_chart(aligned, out_path: str):
    means=[aligned["prior_discount_rate"] or 0, aligned["recent_discount_rate"] or 0]; ns=[aligned["prior_n"], aligned["recent_n"]]
    ci = [(max(0.0, m-0.05), min(1.0, m+0.05)) if (n==0 or m is None) else (m-0.03, m+0.03) for m,n in zip(means, ns)]
    yerr = [[(means[i])-ci[i][0] for i in range(2)], [ci[i][1]-(means[i]) for i in range(2)]]
    x=np.arange(2); plt.figure(); plt.bar(x, means, yerr=yerr, capsize=4)
    plt.xticks(x, ["Prior","Recent"]); plt.ylabel("Avg discount rate"); plt.title(f"Discount rate (window {aligned['window_days']}d)"); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close(); return out_path

def generate_charts(aligned, out_dir: str):
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["repeat_share"] = repeat_share_chart(aligned, str(Path(out_dir)/"repeat_share.png"))
    paths["aov"] = aov_chart(aligned, str(Path(out_dir)/"aov.png"))
    paths["discount_share"] = discount_share_chart(aligned, str(Path(out_dir)/"discount_share.png"))
    return paths
