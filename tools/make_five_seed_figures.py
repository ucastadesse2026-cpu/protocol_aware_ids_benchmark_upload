
import os, csv, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results_final_5seed"
TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

def parse_ms(s):
    if s is None: return (float("nan"), float("nan"))
    s = s.replace("±", "+/-")
    if "+/-" in s:
        a, b = s.split("+/-")
        return (float(a.strip()), float(b.strip()))
    return (float(s.strip()), 0.0)

def load_table6():
    rows = []
    with open(os.path.join(TAB, "table6_robustness_gap.csv")) as f:
        for r in csv.DictReader(f):
            mean_g, sd_g = parse_ms(r.get("G") or r.get("G (pp)") or r.get("G_pp"))
            mean_d, sd_d = parse_ms(r.get("G_domain") or r.get("G_domain (pp)"))
            mean_p, sd_p = parse_ms(r.get("G_protocol") or r.get("G_protocol (pp)"))
            rows.append(dict(model=r["Model"], g=(mean_g, sd_g), d=(mean_d, sd_d), p=(mean_p, sd_p)))
    return rows

def load_table7():
    by = {}
    with open(os.path.join(TAB, "table7_defenses.csv")) as f:
        for r in csv.DictReader(f):
            acc = r.get("Acc S ccsds %") or r.get("Acc_S_ccsds")
            by[(r["Model"], r["Defense"])] = parse_ms(acc)
    return by

def load_table9_confound():
    with open(os.path.join(TAB, "table9_rq_summary.csv")) as f:
        for r in csv.DictReader(f):
            rule = (r.get("Rule") or "").lower()
            if "paa minus clean" in rule or "confound" in rule:
                obs = r.get("Observed") or ""
                out = {}
                for chunk in obs.split(";"):
                    chunk = chunk.strip()
                    m = re.match(r"([a-z_]+):\s*([-0-9.]+)\s*pp\s*\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\]", chunk)
                    if m:
                        model = m.group(1)
                        out[model] = dict(mean=float(m.group(2)), lo=float(m.group(3)), hi=float(m.group(4)))
                return out
    return {}

def fig_robustness_gap():
    data = load_table6()
    models = [r["model"] for r in data]
    d_means = [r["d"][0] for r in data]; d_sds = [r["d"][1] for r in data]
    p_means = [r["p"][0] for r in data]; p_sds = [r["p"][1] for r in data]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w/2, d_means, w, yerr=d_sds, capsize=3, label="G_domain")
    ax.bar(x + w/2, p_means, w, yerr=p_sds, capsize=3, label="G_protocol")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("percentage points")
    ax.set_title("Robustness gap per model (5 seeds, 1,196 windows)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_robustness_gap.png"), dpi=200); plt.close(fig)

def fig_defenses():
    by = load_table7()
    models = ["cnn", "lstm", "random_forest", "transformer"]
    defenses = ["at_paa", "baseline", "paa", "paa_clean_control", "std_at"]
    colors = {"at_paa":"#1f77b4","baseline":"#ff7f0e","paa":"#2ca02c","paa_clean_control":"#d62728","std_at":"#9467bd"}
    x = np.arange(len(models)); w = 0.16
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for i, d in enumerate(defenses):
        means = [by.get((m, d), (float("nan"), 0))[0] for m in models]
        sds   = [by.get((m, d), (float("nan"), 0))[1] for m in models]
        ax.bar(x + (i - 2)*w, means, w, yerr=sds, capsize=2, label=d, color=colors[d])
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("accuracy under conformant attack (%)")
    ax.set_title("Defense effectiveness (5 seeds, 1,196 windows)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_defense_effectiveness.png"), dpi=200); plt.close(fig)

def fig_paa_minus_control():
    conf = load_table9_confound()
    if not conf:
        print("WARN: table9 confound row not found; skipping fig4"); return
    models = ["cnn", "lstm", "random_forest", "transformer"]
    means = [conf[m]["mean"] for m in models]
    lower = [conf[m]["mean"] - conf[m]["lo"] for m in models]
    upper = [conf[m]["hi"] - conf[m]["mean"] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x, means, yerr=[lower, upper], capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("PAA - clean control (pp, Acc S ccsds; 95% interval)")
    ax.set_title("PAA minus clean-augmentation control (5 seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_paa_minus_control.png"), dpi=200); plt.close(fig)

fig_robustness_gap()
fig_defenses()
fig_paa_minus_control()
print("wrote:", os.path.join(FIG, "fig2_robustness_gap.png"))
print("wrote:", os.path.join(FIG, "fig3_defense_effectiveness.png"))
print("wrote:", os.path.join(FIG, "fig4_paa_minus_control.png"))
