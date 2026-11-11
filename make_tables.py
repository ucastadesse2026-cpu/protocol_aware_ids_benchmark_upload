"""Build Tables 3-9 (CSV + Markdown) and Figures 3-5 (PNG) from the CSVs written by experiments/*.py.

    python make_tables.py --config configs/default.yaml

Nothing here invents numbers: every cell is an aggregate (mean +/- sd over seeds x datasets) of a CSV row, and
Table 9 evaluates the decision rules in config `preregistration` against those aggregates.
If ANY input row comes from synthetic data, every output carries a SYNTHETIC-DATA banner/watermark.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config, results_dir
from evaluation.robustness_gap import clean_cross_domain_drop, decompose_gap
from evaluation.transferability import transferability_score

BANNER = ("**SYNTHETIC-DATA BANNER: at least one input row was produced from synthetic data. "
          "These numbers verify the pipeline only and MUST NOT be reported as results.**")


# ----------------------------------------------------------------------------- helpers
def read(cfg, name: str) -> Optional[pd.DataFrame]:
    p = os.path.join(results_dir(cfg, "tables"), name)
    return pd.read_csv(p) if os.path.exists(p) else None


def is_synth(*dfs) -> bool:
    for df in dfs:
        if df is None:
            continue
        for c in ("terr_source", "sat_source"):
            if c in df and (df[c].astype(str) == "synthetic").any():
                return True
    return False


def ci95(x) -> tuple:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return (m - h, m + h)


def ms(x, scale=100.0, nd=2) -> str:
    x = np.asarray([v for v in x if np.isfinite(v)], float) * scale
    if len(x) == 0:
        return "n/a"
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1) if len(x) > 1 else 0.0:.{nd}f}"


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def save(cfg, df: pd.DataFrame, stem: str, title: str, synth: bool, note: str = "") -> None:
    d = results_dir(cfg, "tables")
    df.to_csv(os.path.join(d, stem + ".csv"), index=False)
    with open(os.path.join(d, stem + ".md"), "w", encoding="utf-8") as fh:
        fh.write(f"### {title}\n\n" + (BANNER + "\n\n" if synth else "") + md(df) + ("\n\n" + note if note else "") + "\n")
    print(f"wrote {stem}")


def worst(df, by, col):  # first available row per group
    return df.groupby(by)[col].first()


# ----------------------------------------------------------------------------- tables
def table3(cfg, base, synth):
    if base is None: return
    for dom, stem in (("terrestrial", "table3_clean_terrestrial"), ("satellite", "table3b_clean_satellite_control")):
        t = base[base.domain == dom]
        rows = [dict(Model=m, Dataset=d, **{"Acc %": ms(g.acc), "F1 %": ms(g.f1), "FPR %": ms(g.fpr), "FNR %": ms(g.fnr)})
                for (m, d), g in t.groupby(["model", "dataset"])]
        save(cfg, pd.DataFrame(rows), stem, f"Table 3{'b' if dom == 'satellite' else ''}: clean {dom} detection (mean ± sd)", synth)


def table4(cfg, std, synth):
    if std is None: return
    eps = cfg["attacks"]["standard"]["primary_eps"]
    t = std[np.isclose(std.eps, eps)]
    rows = []
    for (m, a), g in t.groupby(["model", "attack"]):
        gt, gs = g[g.domain == "terrestrial"], g[g.domain == "satellite"]
        rows.append({"Model": m, "Attack": a, "Mode": g["mode"].iloc[0], "Acc T %": ms(gt.acc), "Acc S %": ms(gs.acc),
                     "FNR S %": ms(gs.fnr), "Validator rejection (S) %": ms(gs.validator_rejection)})
    save(cfg, pd.DataFrame(rows), "table4_standard_attacks", f"Table 4: standard attacks at eps={eps}", synth,
         "Validator rejection is a necessary-condition proxy for feature-space examples (lower bound).")


def table5(cfg, cc, synth):
    if cc is None: return
    rows = [{"Model": m, "Attack": a, "Acc S %": ms(g.acc), "Attack success %": ms(g.attack_success_rate),
             "Feasibility %": ms(g.feasibility_rate), "Mean edits": ms(g.mean_edits, 1), "Mean queries": ms(g.mean_queries, 1)}
            for (m, a), g in cc.groupby(["model", "attack"])]
    save(cfg, pd.DataFrame(rows), "table5_ccsds_attacks", "Table 5: CCSDS-conformant attacks", synth)


def gap_per_seed(cfg, base, std, cc) -> Optional[pd.DataFrame]:
    if base is None or std is None or cc is None: return None
    eps, atk = cfg["attacks"]["standard"]["primary_eps"], cfg["gap"]["standard_attack"]
    s = std[(std.attack == atk) & np.isclose(std.eps, eps)]
    out = []
    for (seed, ds, m), g in s.groupby(["seed", "dataset", "model"]):
        c = cc[(cc.seed == seed) & (cc.dataset == ds) & (cc.model == m)]
        kind = next((k for k in cfg["gap"]["ccsds_attack_preference"] if (c.attack == k).any()), None)
        b = base[(base.seed == seed) & (base.dataset == ds) & (base.model == m)]
        if kind is None or b.empty: continue
        a_T = g[g.domain == "terrestrial"].acc.iloc[0]; a_S = g[g.domain == "satellite"].acc.iloc[0]
        a_C = c[c.attack == kind].acc.iloc[0]
        cl = {r.domain: r.acc for r in b.itertuples()}
        out.append(dict(seed=seed, dataset=ds, model=m, ccsds_attack=kind, std_attack=atk, acc_T_std=a_T, acc_S_std=a_S,
                        acc_S_ccsds=a_C, clean_drop=clean_cross_domain_drop(cl["terrestrial"], cl["satellite"]),
                        terr_source=g.terr_source.iloc[0], sat_source=g.sat_source.iloc[0], **decompose_gap(a_T, a_S, a_C)))
    return pd.DataFrame(out)


def table6(cfg, gp, synth, n_sat=None):
    if gp is None or gp.empty: return
    rows = [{"Model": m, "CCSDS attack": g.ccsds_attack.iloc[0], "G": ms(g.G, 1), "G_domain": ms(g.G_domain, 1),
             "G_protocol": ms(g.G_protocol, 1), "G %": ms(g.G_pct, 1), "Clean cross-domain drop": ms(g.clean_drop, 1),
             "Test-sampling ± pp (conservative)": (f"{196 * np.sqrt(g.acc_S_std.mean() * (1 - g.acc_S_std.mean()) / n_sat + g.acc_S_ccsds.mean() * (1 - g.acc_S_ccsds.mean()) / n_sat):.1f}" if n_sat else "n/a")}
            for m, g in gp.groupby("model")]
    save(cfg, pd.DataFrame(rows), "table6_robustness_gap", "Table 6: robustness gap (percentage points)", synth,
         "G = G_domain + G_protocol (exact). Standard attack: " + cfg["gap"]["standard_attack"] + ". "
         "Intervals over seeds reflect model variance only (the satellite test windows are identical for all seeds); the last column is a conservative "
         "binomial half-width for G_protocol from the number of satellite test windows (it ignores the pairing of the two accuracies)." )
    gp.to_csv(os.path.join(results_dir(cfg, "tables"), "gap_per_seed.csv"), index=False)


def table6b(cfg, base, std, cc, synth):
    """EXPLORATORY (post hoc, not pre-registered): G_protocol with the STRONGEST available conformant attack per model."""
    if base is None or std is None or cc is None: return
    eps, atk = cfg["attacks"]["standard"]["primary_eps"], cfg["gap"]["standard_attack"]
    s = std[(std.attack == atk) & np.isclose(std.eps, eps)]
    rows = []
    for m_, gm in s.groupby("model"):
        per = []
        for (seed, ds), g in gm.groupby(["seed", "dataset"]):
            c = cc[(cc.seed == seed) & (cc.dataset == ds) & (cc.model == m_)]
            if c.empty: continue
            a_S = g[g.domain == "satellite"].acc.iloc[0]
            worst = c.loc[c.acc.idxmin()]
            per.append((100 * (a_S - worst.acc), worst.attack))
        if not per: continue
        vals = [v for v, _ in per]; lo, hi = ci95(vals)
        rows.append({"Model": m_, "Strongest conformant attack (most frequent)": max(set(a for _, a in per), key=[a for _, a in per].count),
                     "G_protocol (worst case) pp": f"{np.mean(vals):.1f} [{lo:.1f}, {hi:.1f}]"})
    n = int(cc.n_windows.iloc[0]) if "n_windows" in cc else None
    note = ("EXPLORATORY post-hoc analysis, not pre-registered: the pre-registered G_protocol uses pgd-ccsds (transfer-ccsds for non-differentiable models). "
            "Here the conformant accuracy is the minimum over pgd-, transfer- and greedy-ccsds. ")
    if n:
        note += (f"Satellite test set: {n} windows, the same for every seed, so intervals over seeds do not capture test-set sampling error "
                 f"(a binomial 95% half-width on a single accuracy near 50% is about {100 * 1.96 * np.sqrt(0.25 / n):.1f} pp).")
    save(cfg, pd.DataFrame(rows), "table6b_exploratory_worst_case", "Table 6b: G_protocol with the strongest conformant attack (EXPLORATORY)", synth, note)


def table10(cfg, ind, synth):
    """EXPLORATORY in-domain reference (detectors trained on satellite windows)."""
    if ind is None or ind.empty: return
    base = float(ind.base_rate_attack.mean())
    rows = []
    for m_, g in ind.groupby("model"):
        gp_in = 100 * (g.acc_std - g.acc_ccsds); lo, hi = ci95(gp_in)
        degenerate = bool((g.fpr_clean >= 0.9).mean() >= 0.5 or (g.acc_clean <= max(base, 1 - base) + 0.02).mean() >= 0.5)
        rows.append({"Model": m_, "Acc clean %": ms(g.acc_clean), "FPR clean %": ms(g.fpr_clean), "FNR clean %": ms(g.fnr_clean),
                     "Acc PGD %": ms(g.acc_std), "FNR PGD %": ms(g.fnr_std),
                     "Acc conformant %": ms(g.acc_ccsds), "FNR conformant %": ms(g.fnr_ccsds), "Acc greedy %": ms(g.acc_greedy),
                     "Conformant success %": ms(g.success_ccsds), "G_protocol in-domain pp": f"{gp_in.mean():.1f} [{lo:.1f}, {hi:.1f}]",
                     "Informative?": "no (near-constant or chance-level detector)" if degenerate else "yes (detector above base rate)"})
    # Amendment 2 metadata (view / window / label rule / train attack rate) is present in
    # Run-2C's indomain.csv but not in Run-2's; make the caption fields optional so this
    # table regenerates for both schemas. Numeric content is unchanged.
    def _first(col, default="not recorded"):
        return ind[col].iloc[0] if col in ind.columns else default
    meta_view  = _first("indomain_view")
    meta_win   = _first("window_size")
    meta_rule  = _first("label_rule")
    meta_atk   = (f"{100 * ind.train_attack_rate.mean():.1f}%" if "train_attack_rate" in ind.columns else "not recorded")
    save(cfg, pd.DataFrame(rows), "table10_indomain_reference", "Table 10: in-domain reference, detectors trained on satellite windows (EXPLORATORY)", synth,
         f"EXPLORATORY: added after inspecting earlier runs. Features: {meta_view}, window {meta_win} packets, label rule {meta_rule}, "
         f"attack share of the training windows {meta_atk}. Satellite evaluation subset: {int(ind.n_windows.iloc[0])} windows, {int(ind.n_attacked.iloc[0])} attacks "
         f"(an always-attack predictor scores {100 * base:.1f}%, an always-benign one {100 * (1 - base):.1f}%). Scaler fitted on satellite training windows.")


def defense_frame(cfg, de):
    """Per (dataset, seed, model) deltas against the baseline row."""
    out = []
    for (ds, seed, m), g in de.groupby(["dataset", "seed", "model"]):
        b = g[g.defense == "baseline"]
        if b.empty: continue
        b = b.iloc[0]
        for r in g.itertuples():
            out.append(dict(dataset=ds, seed=seed, model=m, defense=r.defense, acc_T_std=r.acc_T_std, acc_S_ccsds=r.acc_S_ccsds,
                            acc_S_clean=r.acc_S_clean, acc_T_clean=r.acc_T_clean,
                            acc_S_greedy=getattr(r, "acc_S_greedy", np.nan),
                            d_T=r.acc_T_std - b.acc_T_std, d_S=r.acc_S_ccsds - b.acc_S_ccsds,
                            time_mult=r.train_time_s / b.train_time_s if b.train_time_s > 0 else np.nan,
                            clean_drop_pp=100 * (b.acc_T_clean - r.acc_T_clean), latency=r.latency_ms_median, size=r.size_bytes,
                            b_gprot=100 * (b.acc_S_std - b.acc_S_ccsds), terr_source=r.terr_source, sat_source=r.sat_source))
    return pd.DataFrame(out)


def table7(cfg, de, synth):
    if de is None: return None
    df = defense_frame(cfg, de)
    rows = []
    for (m, d), g in df.groupby(["model", "defense"]):
        if d == "baseline":
            tau = "—"
        else:
            base = df[(df.model == m) & (df.defense == "baseline")].set_index(["dataset", "seed"])
            gi = g.set_index(["dataset", "seed"])
            idx = gi.index.intersection(base.index)
            t = transferability_score(gi.loc[idx].acc_S_ccsds, base.loc[idx].acc_S_ccsds, gi.loc[idx].acc_T_std,
                                      base.loc[idx].acc_T_std, seed=0)
            tau = f"{t['tau']:.2f} [{t['ci_low']:.2f}, {t['ci_high']:.2f}]" if t["applicable"] else "n/a (no terrestrial gain)"
        rows.append({"Model": m, "Defense": d, "Acc T std %": ms(g.acc_T_std), "Acc S clean %": ms(g.acc_S_clean),
                     "Acc S ccsds %": ms(g.acc_S_ccsds), "Clean→ccsds drop pp": ms(g.acc_S_clean - g.acc_S_ccsds, 100.0, 1),
                     "Acc S greedy %": ms(g.acc_S_greedy), "tau": tau,
                     "Train-time ×": ms(g.time_mult, 1, 2), "Clean drop pp": ms(g.clean_drop_pp, 1, 2),
                     "Latency ms": ms(g.latency, 1, 3), "Size KB": ms(g["size"] / 1024.0, 1, 1)})
    save(cfg, pd.DataFrame(rows), "table7_defenses", "Table 7: defense effectiveness", synth,
         "paa_clean_control = same satellite windows without adversarial perturbation (separates domain exposure from adversarial content). "
         "tau is computed on accuracy under attack and is CONFOUNDED when clean accuracy itself changes (see Acc S clean and the clean→ccsds drop). "
         "For Random Forests, evaluation uses an independent surrogate; an accuracy under attack ABOVE clean accuracy means the attack is not adaptive and the row is not evidence of robustness.")
    return df


def fed_gap(df):
    g = [decompose_gap(r.acc_T_std, r.acc_S_std, r.acc_S_ccsds) for r in df.itertuples()]
    return pd.DataFrame(g, index=df.index)


def table8(cfg, fed, de, synth):
    if fed is None: return None
    fed = pd.concat([fed, fed_gap(fed)], axis=1)
    ref = None
    if de is not None:
        b = de[(de.defense == "baseline") & (de.model == cfg["federated"]["backbone"])].copy()
        if not b.empty:
            b = pd.concat([b, fed_gap(b)], axis=1)
            ref = b.set_index(["dataset", "seed"]).G
    if ref is not None:
        fed["G_central"] = [ref.get((r.dataset, r.seed), np.nan) for r in fed.itertuples()]
        fed["dG"] = fed.G - fed.G_central
    rows = [{"Variant": v, "K": k, "alpha": a, "Acc S clean %": ms(g.acc_S_clean), "Acc S ccsds %": ms(g.acc_S_ccsds),
             "G": ms(g.G, 1), "G central": ms(g.G_central, 1) if "G_central" in g else "n/a",
             "G fed − central": ms(g.dG, 1) if "dG" in g else "n/a", "Comm MB/round": f"{g.comm_bytes_per_round.mean() / 1e6:.3f}"}
            for (v, k, a), g in fed.groupby(["variant", "k", "alpha"])]
    save(cfg, pd.DataFrame(rows), "table8_federated", "Table 8: federated results", synth)
    return fed


def table9(cfg, gp, dfr, fed, synth):
    P = cfg["preregistration"]
    rows = []
    # RQ1
    obs, outcome = "not computable", "Inconclusive"
    if gp is not None and not gp.empty:
        fam = {m: ci95(g.G_protocol) for m, g in gp.groupby("model")}
        blind = sum(1 for lo, hi in fam.values() if lo > 0); weak = sum(1 for lo, hi in fam.values() if hi < 0)
        obs = f"{len(fam)} families: {blind} with G_protocol CI>0, {weak} with CI<0"
        need = P["rq1_min_families"]
        outcome = ("H-blind supported" if blind >= need else "H-weak supported" if weak >= need
                   else "Inconclusive/mixed" + (" (fewer families than the pre-registered minimum)" if len(fam) < need else ""))
    rows.append({"RQ": "RQ1", "Rule": f"G_protocol CI excludes 0 (same sign) for >= {P['rq1_min_families']} families", "Observed": obs, "Outcome": outcome})
    # RQ2 / RQ3
    obs2, out2, obs3, out3 = "not computable", "Inconclusive", "not computable", "Inconclusive"
    if dfr is not None and not dfr.empty:
        taus = []
        for m, g in dfr[dfr.defense == "std_at"].groupby("model"):
            base = dfr[(dfr.model == m) & (dfr.defense == "baseline")].set_index(["dataset", "seed"]); gi = g.set_index(["dataset", "seed"])
            idx = gi.index.intersection(base.index)
            t = transferability_score(gi.loc[idx].acc_S_ccsds, base.loc[idx].acc_S_ccsds, gi.loc[idx].acc_T_std, base.loc[idx].acc_T_std)
            if t["applicable"]: taus.append(t["tau"])
        if taus:
            low = sum(t < P["rq2_tau_threshold"] for t in taus)
            obs2 = f"{low}/{len(taus)} applicable models with tau < {P['rq2_tau_threshold']}"
            out2 = "Supported (limited transfer)" if low > len(taus) / 2 else "Not supported"
        ok, tot, ctrl = 0, 0, []
        for m, g in dfr[dfr.defense == "paa"].groupby("model"):
            gprot = g.b_gprot.mean()
            if not np.isfinite(gprot) or gprot <= 0: continue
            tot += 1
            rec = 100 * g.d_S.mean() / gprot
            c = dfr[(dfr.model == m) & (dfr.defense == "paa_clean_control")]
            ctrl.append(100 * c.d_S.mean() / gprot if len(c) else np.nan)
            ok += int(rec >= 100 * P["rq3_min_recovery"] and g.time_mult.mean() <= P["rq3_max_time_multiple"] and g.clean_drop_pp.mean() <= P["rq3_clean_acc_tol_pp"])
        if tot:
            obs3 = f"{ok}/{tot} models meet recovery/time/clean-accuracy criteria; mean clean-control recovery {np.nanmean(ctrl):.0f}% of G_protocol"
            out3 = "Supported" if ok > tot / 2 else "Not supported"
        else:
            obs3 = "G_protocol <= 0 for all models: recovery fraction undefined"
    rows.append({"RQ": "RQ2", "Rule": f"tau < {P['rq2_tau_threshold']} for a majority of models", "Observed": obs2, "Outcome": out2})
    rows.append({"RQ": "RQ3", "Rule": f"PAA recovers >= {100 * P['rq3_min_recovery']:.0f}% of G_protocol at <= {P['rq3_max_time_multiple']}x training time and <= {P['rq3_clean_acc_tol_pp']} pp clean loss",
                 "Observed": obs3, "Outcome": out3})
    # RQ4
    obs4, out4 = "not computable", "Inconclusive"
    if fed is not None and "dG" in fed and fed.dG.notna().any():
        # Both federated variants (fedavg and federated_paa) are pooled, matching Section VI-G
        # of the manuscript: "the mean of the absolute per-seed differences across the twenty
        # seed-variant-dataset pairs is 24.31 pp".
        f0 = fed
        hi = f0[f0.alpha >= 0.5].dG.abs().mean(); lo = f0[f0.alpha < 0.5].dG.abs().mean()
        if np.isfinite(lo):
            obs4 = f"mean |G_fed − G_central|: {hi:.1f} pp (alpha>=0.5), {lo:.1f} pp (alpha<0.5)"
            out4 = "Supported" if hi <= P["rq4_tolerance_pp"] and lo > hi else "Not supported"
        else:
            obs4 = f"mean |G_fed − G_central|: {hi:.1f} pp (alpha>=0.5), not run (alpha<0.5)"
            out4 = "Not supported (alpha = 0.5 only; alpha = 0.1 not run)"
    if dfr is not None and not dfr.empty:
        parts = []
        for mm, g in dfr[dfr.defense.isin(["paa", "paa_clean_control"])].groupby("model"):
            a = g[g.defense == "paa"].set_index(["dataset", "seed"]).acc_S_ccsds; b = g[g.defense == "paa_clean_control"].set_index(["dataset", "seed"]).acc_S_ccsds
            idx = a.index.intersection(b.index)
            if len(idx) > 1:
                d = 100 * (a.loc[idx] - b.loc[idx]).to_numpy(); lo, hi = ci95(d)
                parts.append(f"{mm}: {d.mean():.1f} pp [{lo:.1f}, {hi:.1f}]")
        rows.append({"RQ": "RQ3 (confound check, informational)", "Rule": "PAA minus clean-augmentation control in Acc S ccsds; interval covering 0 = no evidence that adversarial content adds anything beyond satellite-domain exposure",
                     "Observed": "; ".join(parts) or "not computable", "Outcome": "informational (no pre-registered threshold)"})
    rows.append({"RQ": "RQ4", "Rule": f"|G_fed − G_central| <= {P['rq4_tolerance_pp']} pp for alpha >= 0.5 and larger at alpha = 0.1", "Observed": obs4, "Outcome": out4})
    save(cfg, pd.DataFrame(rows), "table9_rq_summary", "Table 9: research questions, pre-registered rules and measured outcomes", synth)


# ----------------------------------------------------------------------------- figures
def figures(cfg, gp, dfr, synth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = results_dir(cfg, "figures")

    def wm(ax):
        if synth:
            ax.text(0.5, 0.5, "SYNTHETIC DATA", transform=ax.transAxes, ha="center", va="center", fontsize=28, color="red", alpha=0.25, rotation=25)

    if gp is not None and not gp.empty:
        g = gp.groupby("model")[["G_domain", "G_protocol"]].agg(["mean", "std"]).fillna(0.0)
        fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(len(g)); w = 0.38
        ax.bar(x - w / 2, g["G_domain"]["mean"], w, yerr=g["G_domain"]["std"], label="G_domain", capsize=3)
        ax.bar(x + w / 2, g["G_protocol"]["mean"], w, yerr=g["G_protocol"]["std"], label="G_protocol", capsize=3)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20); ax.set_ylabel("percentage points")
        ax.set_title("Robustness gap per model (mean ± sd over seeds)"); ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8); wm(ax); fig.tight_layout()
        fig.savefig(os.path.join(fdir, "fig3_robustness_gap.png"), dpi=200); plt.close(fig)
    if dfr is not None and not dfr.empty:
        pv = dfr.groupby(["model", "defense"]).acc_S_ccsds.mean().unstack() * 100
        fig, ax = plt.subplots(figsize=(7, 3.6)); pv.plot(kind="bar", ax=ax, rot=20)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax.set_ylabel("accuracy under conformant attack (%)"); ax.set_title("Defense effectiveness"); wm(ax); fig.tight_layout()
        fig.savefig(os.path.join(fdir, "fig4_defense_effectiveness.png"), dpi=200); plt.close(fig)
        s = dfr.groupby(["model", "defense"]).agg(t=("time_mult", "mean"), a=("acc_S_ccsds", "mean"), sz=("size", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for d, g in s.groupby("defense"):
            ax.scatter(g.t, g.a * 100, s=30 + 200 * g.sz / max(s.sz.max(), 1), label=d, alpha=0.7)
        ax.set_xlabel("training time (× baseline)"); ax.set_ylabel("accuracy under conformant attack (%)")
        ax.set_title("Overhead vs robustness (marker area ∝ model size)"); ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7); wm(ax); fig.tight_layout()
        fig.savefig(os.path.join(fdir, "fig5_overhead_vs_robustness.png"), dpi=200); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out-dir", help="override output.dir (e.g. the folder written by tools/merge_results.py)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg.setdefault("output", {})["dir"] = args.out_dir
    base, std, cc = read(cfg, "baseline_clean.csv"), read(cfg, "standard_attacks.csv"), read(cfg, "ccsds_attacks.csv")
    de, fed = read(cfg, "defenses.csv"), read(cfg, "federated.csv")
    synth = is_synth(base, std, cc, de, fed, read(cfg, "indomain.csv"))
    if synth:
        print("WARNING: synthetic data detected -> outputs are stamped SYNTHETIC-DATA")
    table3(cfg, base, synth); table4(cfg, std, synth); table5(cfg, cc, synth)
    gp = gap_per_seed(cfg, base, std, cc)
    table6(cfg, gp, synth, int(cc.n_windows.iloc[0]) if cc is not None and "n_windows" in cc else None)
    table6b(cfg, base, std, cc, synth)
    table10(cfg, read(cfg, "indomain.csv"), synth)
    dfr = table7(cfg, de, synth)
    fed2 = table8(cfg, fed, de, synth)
    table9(cfg, gp, dfr, fed2, synth)
    figures(cfg, gp, dfr, synth)


if __name__ == "__main__":
    main()
