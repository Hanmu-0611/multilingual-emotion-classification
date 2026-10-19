#!/usr/bin/env python3
"""Run-level analysis of the fixed Study 1 re-run (final-epoch protocol, 10 seeds).

Usage: python3 analyze_study1_fixed.py <results_dir> <out_dir>
  results_dir: folder with run_level_results.csv, source_level_results.csv, emotion_level_results.csv

Design of the Model-Strategy comparison (reviewer 2 #4, #7):
  TargetLanguage {KO, ZH, EN}  x  Model {multilingual mBERT, language-specific}  x  SourceLanguage {KO, ZH, EN}
  * Unit of replication = one training run (10 seeds per Target x Model cell, 60 runs).
  * Every run is evaluated on 500 test sentences per source language, so SourceLanguage is a WITHIN-run factor:
    the 3 source-level F1 values of one run are not independent. This is analysed as a split-plot (mixed) ANOVA:
      between runs : Target, Model, Target x Model     -> error = runs within cells (df 54)
      within runs  : Source, Source x Target, Source x Model, Source x Target x Model -> error = Source x run (df 108)
  * The Original (mixed-language) dataset has no language-specific model; it is a reference condition (mBERT only), reported
    separately and NOT part of the Model-Strategy ANOVA.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
TARGETS = ["KO", "ZH", "EN"]
STRATS = ["multilingual", "language_specific"]
SOURCES = ["ko", "zh", "en"]
pd.set_option("display.width", 200)


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    m = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def welch_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    diffs = RNG.choice(a, (10000, len(a))).mean(1) - RNG.choice(b, (10000, len(b))).mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(mean_a=a.mean(), mean_b=b.mean(), diff=a.mean() - b.mean(), ci_lo=lo, ci_hi=hi, welch_t=t, welch_p=p,
                cohens_d=(a.mean() - b.mean()) / sp, mannwhitney_p=pu)


run = pd.read_csv(RES / "run_level_results.csv")
src = pd.read_csv(RES / "source_level_results.csv")
emo = pd.read_csv(RES / "emotion_level_results.csv")
seeds = sorted(run.seed.unique())
print(f"seeds={seeds}  runs={len(run)}")
assert len(run) == len(seeds) * 7

run["run_id"] = run.apply(lambda r: f"{r.seed}_{r.target}_{r.strategy}", axis=1)
src["run_id"] = src.apply(lambda r: f"{r.seed}_{r.target}_{r.strategy}", axis=1)
cmp_run = run[run.target.isin(TARGETS)].copy()
cmp_src = src[src.target.isin(TARGETS)].copy()
assert len(cmp_run) == 60 and len(cmp_src) == 180

summary = {"n_seeds": len(seeds), "seeds": [int(s) for s in seeds]}

# ---------------------------------------------------------------- [1] descriptives
print("\n[1] run-level Macro-F1 (overall 1500 test sentences), mean ± SD [95% bootstrap CI over seeds]")
rows = []
for (t, s), g in run.groupby(["target", "strategy"]):
    lo, hi = boot_ci(g.macro_f1)
    rows.append(dict(target=t, strategy=s, n=len(g), mean=g.macro_f1.mean(), sd=g.macro_f1.std(ddof=1), ci_lo=lo, ci_hi=hi,
                     min=g.macro_f1.min(), max=g.macro_f1.max(), mean_acc=g.accuracy.mean()))
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "table_run_level_cell_means.csv", index=False)
print(t1.round(3).to_string(index=False))

print("\n    per source language (source-level F1, 500 test sentences each), mean ± SD over seeds")
rows = []
for (t, s, so), g in src.groupby(["target", "strategy", "source_language"]):
    lo, hi = boot_ci(g.macro_f1)
    rows.append(dict(target=t, strategy=s, source_language=so, n=len(g), mean=g.macro_f1.mean(), sd=g.macro_f1.std(ddof=1), ci_lo=lo, ci_hi=hi))
t2 = pd.DataFrame(rows)
t2.to_csv(OUT / "table_source_level_cell_means.csv", index=False)
piv = t2.assign(txt=lambda d: d["mean"].round(3).astype(str) + " ± " + d["sd"].round(3).astype(str)).pivot_table(
    index=["target", "strategy"], columns="source_language", values="txt", aggfunc="first")[SOURCES]
print(piv.to_string())

# ---------------------------------------------------------------- [2] run-level 2-way ANOVA
print("\n[2] run-level ANOVA: Target(3) x Model(2), 60 runs, 10 per cell  (the unit of analysis is a run)")
m_run = smf.ols("macro_f1 ~ C(target) * C(strategy)", data=cmp_run).fit()
a_run = sm.stats.anova_lm(m_run, typ=2)
ss_tot = a_run["sum_sq"].sum()
a_run["eta_sq"] = a_run["sum_sq"] / ss_tot
a_run["partial_eta_sq"] = a_run["sum_sq"] / (a_run["sum_sq"] + a_run.loc["Residual", "sum_sq"])
print(a_run.round(4).to_string())
a_run.to_csv(OUT / "anova_run_level_target_x_model.csv")
lev = stats.levene(*[g.macro_f1.values for _, g in cmp_run.groupby(["target", "strategy"])])
sh = stats.shapiro(m_run.resid)
print(f"    assumptions: Levene W={lev.statistic:.2f} p={lev.pvalue:.3g} | Shapiro(resid) W={sh.statistic:.3f} p={sh.pvalue:.3g}")
rk = cmp_run.assign(r=cmp_run.macro_f1.rank())
a_rk = sm.stats.anova_lm(smf.ols("r ~ C(target) * C(strategy)", data=rk).fit(), typ=2)
print("    robustness: rank-transformed ANOVA p-values:", {k: float(f"{v:.3g}") for k, v in a_rk["PR(>F)"].dropna().items()})
summary["run_level_anova"] = a_run.fillna(0).to_dict()
summary["assumptions_run_level"] = dict(levene_p=float(lev.pvalue), shapiro_p=float(sh.pvalue),
                                        rank_anova_p={k: float(v) for k, v in a_rk["PR(>F)"].dropna().items()})

# ---------------------------------------------------------------- [3] split-plot ANOVA with Source within run
print("\n[3] split-plot ANOVA on source-level F1 (180 values = 60 runs x 3 source languages)")
fx = smf.ols("macro_f1 ~ C(target) * C(strategy) * C(source_language)", data=cmp_src).fit()
fa = sm.stats.anova_lm(fx, typ=2)
cell = cmp_src.groupby(["target", "strategy"]).macro_f1.transform("mean")
runmean = cmp_src.groupby("run_id").macro_f1.transform("mean")
cellmean_run = cmp_src.assign(c=cell).groupby("run_id").c.transform("mean")
ss_runs = 3 * ((cmp_src.groupby("run_id").macro_f1.mean() - cmp_src.assign(c=cell).groupby("run_id").c.mean()) ** 2).sum()
df_runs = 60 - 6
ss_within_err = fa.loc["Residual", "sum_sq"] - ss_runs
df_within_err = fa.loc["Residual", "df"] - df_runs
rows = []
for term in fa.index[:-1]:
    between = "source_language" not in term
    err_ss, err_df = (ss_runs, df_runs) if between else (ss_within_err, df_within_err)
    ss, df = fa.loc[term, "sum_sq"], fa.loc[term, "df"]
    F = (ss / df) / (err_ss / err_df)
    rows.append(dict(term=term, stratum="between runs" if between else "within runs", df=df, df_error=err_df, SS=ss, F=F,
                     p=stats.f.sf(F, df, err_df), eta_sq=ss / fa["sum_sq"].sum(), partial_eta_sq=ss / (ss + err_ss)))
sp = pd.DataFrame(rows)
sp.to_csv(OUT / "anova_split_plot_target_model_source.csv", index=False)
print(sp.round(4).to_string(index=False))
# naive version (what the submitted paper's design implicitly did): treats all 180 values as independent
naive = fa.assign(eta_sq=fa["sum_sq"] / fa["sum_sq"].sum())
naive.to_csv(OUT / "anova_NAIVE_independent_rows_for_comparison.csv")
# mixed model (random intercept per run) as a cross-check on the fixed-effect pattern
try:
    mm = smf.mixedlm("macro_f1 ~ C(target) * C(strategy) * C(source_language)", cmp_src, groups=cmp_src["run_id"]).fit(reml=True)
    summary["mixedlm_run_intercept_var"] = float(mm.cov_re.iloc[0, 0])
    summary["mixedlm_resid_var"] = float(mm.scale)
    icc = float(mm.cov_re.iloc[0, 0] / (mm.cov_re.iloc[0, 0] + mm.scale))
    print(f"    mixed-model cross-check: run-intercept variance={mm.cov_re.iloc[0,0]:.5f}, residual={mm.scale:.5f}, ICC={icc:.3f}")
    summary["icc_run"] = icc
except Exception as exc:  # noqa: BLE001
    print("    mixed model failed:", exc)

# ---------------------------------------------------------------- [4] strategy contrasts
print("\n[4] language-specific minus multilingual (run-level Macro-F1), per target language")
rows = []
for t in TARGETS:
    a = cmp_run[(cmp_run.target == t) & (cmp_run.strategy == "language_specific")].macro_f1
    b = cmp_run[(cmp_run.target == t) & (cmp_run.strategy == "multilingual")].macro_f1
    rows.append(dict(target=t, **welch_d(a, b)))
pooled = welch_d(cmp_run[cmp_run.strategy == "language_specific"].macro_f1, cmp_run[cmp_run.strategy == "multilingual"].macro_f1)
rows.append(dict(target="ALL", **pooled))
t4 = pd.DataFrame(rows)
t4.to_csv(OUT / "contrast_language_specific_vs_mbert.csv", index=False)
print(t4.round(4).to_string(index=False))
print("    (Holm-adjusted p for the three per-language contrasts:",
      np.round(sm.stats.multipletests(t4.welch_p[:3], method="holm")[1], 5).tolist(), ")")

print("\n    Original (mixed-language) reference vs mBERT on a single-language dataset")
orig = run[run.target == "Original"].macro_f1
rows = []
for t in TARGETS:
    b = cmp_run[(cmp_run.target == t) & (cmp_run.strategy == "multilingual")].macro_f1
    rows.append(dict(contrast=f"Original - {t}(mBERT)", **welch_d(orig, b)))
t5 = pd.DataFrame(rows)
t5.to_csv(OUT / "contrast_original_vs_single_language_mbert.csv", index=False)
print(t5.round(4).to_string(index=False))

# ---------------------------------------------------------------- [5] source-language profile and emotions
print("\n[5] emotion-level F1, mean over seeds (rows: target/strategy/source; mean of 10 seeds)")
e = emo[emo.target.isin(TARGETS)].groupby(["target", "strategy", "source_language", "emotion"]).f1.agg(["mean", "std"]).reset_index()
e.to_csv(OUT / "table_emotion_f1_mean_sd.csv", index=False)
em = emo[emo.target.isin(TARGETS)].groupby("emotion").f1.agg(["mean", "std"]).round(3)
print(em.to_string())
perseed = emo[emo.target.isin(TARGETS)].groupby(["seed", "target", "strategy", "source_language", "emotion"]).f1.mean().reset_index()
wide = perseed.groupby(["seed", "emotion"]).f1.mean().unstack()
fr = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
print(f"    Friedman test across the 5 emotions (blocks = seeds, F1 averaged over configs): chi2={fr.statistic:.2f}, p={fr.pvalue:.3g}")
summary["friedman_emotions"] = dict(chi2=float(fr.statistic), p=float(fr.pvalue))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(6.5, 4))
w = 0.35
for i, s in enumerate(STRATS):
    sub = t1[(t1.strategy == s) & t1.target.isin(TARGETS)].set_index("target").loc[TARGETS]
    ax.bar(np.arange(3) + (i - 0.5) * w, sub["mean"], w, yerr=[sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]], capsize=3,
           label={"multilingual": "mBERT (multilingual)", "language_specific": "language-specific"}[s])
ax.axhline(t1[t1.target == "Original"]["mean"].iloc[0], ls="--", c="gray", lw=1, label="Original, mBERT (reference)")
ax.set_xticks(range(3)); ax.set_xticklabels(TARGETS); ax.set_ylabel("Macro-F1 (mean, 95% CI over 10 seeds)")
ax.set_xlabel("Target language dataset"); ax.set_ylim(0.5, 0.75); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_strategy_by_target.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, s in zip(axes, STRATS):
    for t in TARGETS:
        sub = t2[(t2.strategy == s) & (t2.target == t)].set_index("source_language").loc[SOURCES]
        ax.errorbar([x.upper() for x in SOURCES], sub["mean"], yerr=[sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]], marker="o", capsize=3, label=f"target {t}")
    ax.set_title("mBERT (multilingual)" if s == "multilingual" else "language-specific"); ax.set_xlabel("Source language of test sentences")
axes[0].set_ylabel("Macro-F1 (mean, 95% CI over seeds)"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_source_language_profile.png", dpi=200); plt.close(fig)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(f"\nsaved to {OUT}")
