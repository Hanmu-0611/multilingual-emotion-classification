"""Analysis of the corrected Study 2 (final-epoch protocol, disjoint train/test pools, 10 seeds).

usage: python3 analyze_new_study2.py <results_dir> <out_dir>
  <results_dir> must contain run_level_results.csv and emotion_level_results.csv (outputs/fixed_final_epoch_10seeds)

Every cell (Model x Source x Target) now has one run per seed, so the factorial ANOVA has real replicates
(residual df = 36*(n_seeds-1)) instead of the pseudo-replicated emotion-level rows used in the original Table 4.
Only complete seeds (36 runs) are analysed.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
LANGS = ["KO", "ZH", "EN"]
MODELS = ["bert-base-multilingual-cased", "klue/roberta-base", "hfl/chinese-roberta-wwm-ext", "roberta-base"]
SHORT = {"bert-base-multilingual-cased": "mBERT", "klue/roberta-base": "KLUE-RoBERTa",
         "hfl/chinese-roberta-wwm-ext": "Chinese-RoBERTa", "roberta-base": "RoBERTa"}
EMO = ["anger", "fear", "joy", "sadness", "surprise"]
rng = np.random.default_rng(2026)
summary = {}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return (m - h, m + h)


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(b, [2.5, 97.5]))


# ------------------------------------------------------------------ load
run = pd.read_csv(RES / "run_level_results.csv")
met = "f1_last" if "f1_last" in run.columns else "macro_f1"
run["f1"] = run[met]
for c in ["source", "target"]:
    run[c] = run[c].str.upper()
cnt = run.groupby("seed").size()
complete = cnt[cnt == 36].index.tolist()
run = run[run.seed.isin(complete)].copy()
n_seeds = len(complete)
summary.update({"metric": met, "complete_seeds": complete, "n_seeds": n_seeds, "n_runs": int(len(run))})
print(f"metric={met}  complete seeds={complete}  runs={len(run)}")

emo = pd.read_csv(RES / "emotion_level_results.csv")
emo["source"] = emo.source.str.upper(); emo["target"] = emo.target.str.upper()
emo = emo[emo.seed.isin(complete)].copy()

# ------------------------------------------------------------------ 1. performance matrix (reviewer 2 #12)
cell = run.groupby(["model", "source", "target"]).f1.agg(["mean", "std", "count"]).reset_index()
cell[["ci_lo", "ci_hi"]] = [ci95(run[(run.model == r.model) & (run.source == r.source) & (run.target == r.target)].f1) for r in cell.itertuples()]
cell.round(4).to_csv(OUT / "table_cell_means_sd_ci.csv", index=False)
cell["txt"] = cell["mean"].map("{:.3f}".format) + " ± " + cell["std"].map("{:.3f}".format)
mat = cell.pivot_table(index=["model", "source"], columns="target", values="txt", aggfunc="first")[LANGS]
mat.to_csv(OUT / "table_performance_matrix_mean_sd.csv")
print("\n[1] performance matrix (mean ± SD over seeds)\n", mat.to_string())

# ------------------------------------------------------------------ 2. factorial ANOVA with true replicates (R2 #6, #8)
def anova_table(df, formula):
    m = ols(formula, data=df).fit()
    a = sm.stats.anova_lm(m, typ=2)
    ss_res = a.loc["Residual", "sum_sq"]
    a["eta_sq"] = a["sum_sq"] / a["sum_sq"].sum()
    a["partial_eta_sq"] = a["sum_sq"] / (a["sum_sq"] + ss_res)
    return a, m

if n_seeds >= 2:
    a, m = anova_table(run, "f1 ~ C(model)*C(source)*C(target)")
    a.round(5).to_csv(OUT / "anova_full_factorial_all_models.csv")
    print(f"\n[2] Type-II ANOVA, all models, {len(run)} runs, residual df={int(a.loc['Residual','df'])}\n", a[["df", "F", "PR(>F)", "eta_sq", "partial_eta_sq"]].round(4).to_string())
    summary["anova_all_models"] = {k: {"F": float(v["F"]), "p": float(v["PR(>F)"]), "eta_sq": float(v["eta_sq"]), "partial_eta_sq": float(v["partial_eta_sq"])}
                                   for k, v in a.drop(index="Residual").iterrows()}
    # assumptions
    res = m.resid
    lev = stats.levene(*[g.f1.values for _, g in run.groupby(["model", "source", "target"])])
    sw = stats.shapiro(res.sample(min(len(res), 4000), random_state=1))
    summary["assumptions"] = {"levene_W": float(lev.statistic), "levene_p": float(lev.pvalue), "shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue)}
    print(f"   assumptions: Levene W={lev.statistic:.2f} p={lev.pvalue:.3g} | Shapiro (residuals) W={sw.statistic:.3f} p={sw.pvalue:.3g}")
    # sensitivity: the single multilingual model (cleanest cross-lingual test)
    mb = run[run.model == "bert-base-multilingual-cased"]
    a2, _ = anova_table(mb, "f1 ~ C(source)*C(target)")
    a2.round(5).to_csv(OUT / "anova_mBERT_only.csv")
    print(f"\n   mBERT only ({len(mb)} runs, residual df={int(a2.loc['Residual','df'])})\n", a2[["df", "F", "PR(>F)", "eta_sq", "partial_eta_sq"]].round(4).to_string())
    summary["anova_mBERT"] = {k: {"F": float(v["F"]), "p": float(v["PR(>F)"]), "partial_eta_sq": float(v["partial_eta_sq"])} for k, v in a2.drop(index="Residual").iterrows()}

# ------------------------------------------------------------------ 3. same-language baseline vs cross-lingual (advisor's suggestion)
base = run[run.source == run.target][["seed", "model", "source", "f1"]].rename(columns={"f1": "f1_same"})
dd = run[run.source != run.target].merge(base, on=["seed", "model", "source"])
dd["delta"] = dd.f1 - dd.f1_same
rows = []
for (s, t), g in dd.groupby(["source", "target"]):
    per_seed = g.groupby("seed").delta.mean()          # average over models within seed -> seeds are the replicates
    lo, hi = boot_ci(per_seed)
    rows.append({"source": s, "target": t, "mean_delta": per_seed.mean(), "ci_lo": lo, "ci_hi": hi, "n_seeds": len(per_seed)})
drop = pd.DataFrame(rows); drop.round(4).to_csv(OUT / "cross_lingual_drop_vs_same_language.csv", index=False)
print("\n[3] cross-lingual change relative to same-language baseline (mean over models; 95% bootstrap CI over seeds)\n", drop.round(3).to_string(index=False))
mb_drop = dd[dd.model == "bert-base-multilingual-cased"].groupby(["source", "target"]).delta.agg(["mean", "std"]).round(3)
mb_drop.to_csv(OUT / "cross_lingual_drop_mBERT_only.csv")
summary["same_language_mean_f1"] = float(base.f1_same.mean()); summary["cross_lingual_mean_f1"] = float(dd.f1.mean())

# ------------------------------------------------------------------ 4. emotion-level variability (R2 #10, R3 #5)
pm = emo.groupby(["seed", "emotion", "source", "target"]).f1.mean().reset_index()   # average over models
var_rows = []
for (sd, e), g in pm.groupby(["seed", "emotion"]):
    path_means = g.groupby(["source", "target"]).f1.mean().values
    var_rows.append({"seed": sd, "emotion": e, "sd_across_paths": path_means.std(ddof=1), "mean_f1": g.f1.mean()})
ev = pd.DataFrame(var_rows)
tab = ev.groupby("emotion").agg(mean_f1=("mean_f1", "mean"), sd_across_paths_mean=("sd_across_paths", "mean"), sd_across_paths_sd=("sd_across_paths", "std"))
tab[["ci_lo", "ci_hi"]] = [boot_ci(ev[ev.emotion == e].sd_across_paths) for e in tab.index]
tab.round(4).to_csv(OUT / "emotion_variability_across_paths.csv")
print("\n[4] emotion F1 variability: SD of the 9 path means (model-averaged), summarised over seeds\n", tab.round(3).to_string())
if n_seeds >= 3:
    wide = ev.pivot(index="seed", columns="emotion", values="sd_across_paths")
    fr = stats.friedmanchisquare(*[wide[e].values for e in EMO])
    print(f"   Friedman test across the 5 emotions on that SD (blocks=seeds): chi2={fr.statistic:.2f}, p={fr.pvalue:.4f}")
    summary["emotion_variability_friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}
    top = wide.idxmax(axis=1).value_counts().to_dict(); summary["most_variable_emotion_per_seed"] = top
    print("   most variable emotion per seed:", top)
emo_path = emo.groupby(["emotion", "source", "target"]).f1.agg(["mean", "std"]).round(4)
emo_path.to_csv(OUT / "emotion_f1_by_path.csv")

# ------------------------------------------------------------------ 5. figures (reviewer 2 minor #10)
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for k, mname in enumerate(MODELS):
    z = run[run.model == mname].pivot_table(index="source", columns="target", values="f1", aggfunc="mean").reindex(index=LANGS, columns=LANGS)
    im = ax[k].imshow(z.values, vmin=0, vmax=max(0.9, np.nanmax(z.values)), cmap="viridis")
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(LANGS); ax[k].set_yticks(range(3)); ax[k].set_yticklabels(LANGS)
    ax[k].set_xlabel("Target (test) language"); ax[k].set_ylabel("Source (train) language" if k == 0 else "")
    ax[k].set_title(SHORT[mname])
    for i in range(3):
        for j in range(3):
            ax[k].text(j, i, f"{z.values[i, j]:.2f}", ha="center", va="center", color="w" if z.values[i, j] < 0.5 else "k", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.012, label="Macro-F1 (mean over seeds)")
fig.savefig(OUT / "fig_heatmap_source_target_by_model.png", dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for k, (title, dfp) in enumerate([("All four models", run), ("mBERT only", run[run.model == "bert-base-multilingual-cased"])]):
    for s, col in zip(LANGS, ["#1b9e77", "#d95f02", "#7570b3"]):
        g = dfp[dfp.source == s]
        mu = [g[g.target == t].f1.mean() for t in LANGS]
        err = [(ci95(g[g.target == t].f1)[1] - ci95(g[g.target == t].f1)[0]) / 2 if len(g[g.target == t]) > 1 else 0 for t in LANGS]
        ax[k].errorbar(range(3), mu, yerr=err, marker="o", capsize=3, label=f"Source={s}", color=col)
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(LANGS); ax[k].set_xlabel("Target (test) language"); ax[k].set_title(title)
ax[0].set_ylabel("Macro-F1 (mean, 95% CI)"); ax[0].legend(frameon=False)
fig.savefig(OUT / "fig_interaction_source_target.png", dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
lab = [f"{r.source}→{r.target}" for r in drop.itertuples()]
ax.bar(lab, drop.mean_delta, yerr=[drop.mean_delta - drop.ci_lo, drop.ci_hi - drop.mean_delta], capsize=3, color="#4c72b0")
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Δ Macro-F1 vs same-language (Source→Source)"); ax.set_title("Cross-lingual cost relative to same-language baseline")
fig.savefig(OUT / "fig_cross_lingual_drop.png", dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
em = emo.groupby(["seed", "emotion"]).f1.mean().reset_index()
ax.boxplot([em[em.emotion == e].f1.values for e in EMO]); ax.set_xticks(range(1, len(EMO) + 1)); ax.set_xticklabels(EMO); ax.set_ylabel("Emotion F1 (mean over all 36 conditions)"); ax.set_title("Emotion-specific F1 across seeds")
fig.savefig(OUT / "fig_emotion_f1_over_seeds.png", dpi=200, bbox_inches="tight"); plt.close(fig)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
print("\nsaved to", OUT)
