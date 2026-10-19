#!/usr/bin/env python3
"""Study 3 (emotion stability across language conditions) recomputed on the fixed 10-seed data (reviewer 2 #10, reviewer 3 #5).

Usage: python3 analyze_study3_fixed.py <study2_results_dir> <study1_results_dir> <out_dir>

Definition kept from the submitted manuscript (so numbers stay comparable):
    stability SD of emotion e = SD (ddof=1) of the 6 marginal means
        [mean F1 of e for each of the 3 source languages] + [mean F1 of e for each of the 3 target languages]
What is new: it is computed separately for every training seed (10 values per emotion) instead of once, so the ranking
comes with spread, a rank-consistency count, a Friedman test across emotions (blocks = seeds) and Holm-corrected Wilcoxon
contrasts. The seeds only vary training randomness (data split fixed), which is stated in the report.

Study3B = Study 2 (Source->Target, 4 models): reported for mBERT only (the same model in every cell) and for all 4 models.
Study3A = Study 1 (Target x Model x Source on the shared test sets): 'source' = origin of the test sentences, 'target' = dataset language.
Also: split-plot ANOVA with Emotion as a within-run factor on mBERT Study 2 runs (run = Source x Target x seed).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

S2, S1, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
EMO = ["anger", "fear", "joy", "sadness", "surprise"]
pd.set_option("display.width", 200)
MBERT = "bert-base-multilingual-cased"
summary = {}


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    m = RNG.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def stability_per_seed(df, src_col, tgt_col):
    """df: seed, emotion, f1, src_col, tgt_col (any other factors are averaged out)."""
    rows = []
    for seed, g in df.groupby("seed"):
        for e in EMO:
            p = g[g.emotion == e]
            vals = np.concatenate([p.groupby(src_col).f1.mean().values, p.groupby(tgt_col).f1.mean().values])
            assert len(vals) == 6
            rows.append(dict(seed=seed, emotion=e, mean_f1=vals.mean(), sd=vals.std(ddof=1), range=vals.max() - vals.min()))
    return pd.DataFrame(rows)


def report(label, per_seed, key):
    print(f"\n--- {label} ---")
    g = per_seed.groupby("emotion")
    t = pd.DataFrame({"mean_f1": g.mean_f1.mean(), "stability_sd_mean": g.sd.mean(), "stability_sd_sd_over_seeds": g.sd.std(ddof=1)})
    cis = {e: boot_ci(per_seed[per_seed.emotion == e].sd) for e in EMO}
    t["ci_lo"] = [cis[e][0] for e in t.index]
    t["ci_hi"] = [cis[e][1] for e in t.index]
    wide = per_seed.pivot(index="seed", columns="emotion", values="sd")[EMO]
    t["seeds_rank_most_variable"] = (wide.idxmax(axis=1).value_counts()).reindex(t.index).fillna(0).astype(int)
    t["seeds_rank_most_stable"] = (wide.idxmin(axis=1).value_counts()).reindex(t.index).fillna(0).astype(int)
    t["mean_rank_(1=most variable)"] = wide.rank(axis=1, ascending=False).mean().reindex(t.index)
    t = t.sort_values("stability_sd_mean", ascending=False)
    print(t.round(3).to_string())
    fr = stats.friedmanchisquare(*[wide[e].values for e in EMO])
    n, k = wide.shape[0], len(EMO)
    kendall_w = fr.statistic / (n * (k - 1))
    print(f"Friedman across emotions (blocks=seeds): chi2={fr.statistic:.2f}, df={k-1}, p={fr.pvalue:.3g}, Kendall's W={kendall_w:.3f}")
    pairs, ps = [], []
    for a, b in combinations(EMO, 2):
        w = stats.wilcoxon(wide[a], wide[b])
        pairs.append((a, b, float((wide[a] - wide[b]).mean()), float(w.pvalue)))
        ps.append(w.pvalue)
    adj = sm.stats.multipletests(ps, method="holm")[1]
    pw = pd.DataFrame(pairs, columns=["emotion_a", "emotion_b", "mean_diff_sd_a_minus_b", "wilcoxon_p"])
    pw["holm_p"] = adj
    pw["significant_holm_0.05"] = pw.holm_p < 0.05
    print(f"pairwise Wilcoxon (10 seeds -> smallest attainable two-sided p = {2/2**10:.4f}); Holm-significant pairs: {int(pw['significant_holm_0.05'].sum())}/10")
    print(pw[pw["significant_holm_0.05"]].round(4).to_string(index=False) or "  none")
    t.to_csv(OUT / f"{key}_stability_ranking.csv")
    per_seed.to_csv(OUT / f"{key}_stability_per_seed.csv", index=False)
    pw.to_csv(OUT / f"{key}_pairwise_wilcoxon.csv", index=False)
    summary[key] = dict(friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue), kendall_w=float(kendall_w),
                        most_variable_by_mean=t.index[0], least_variable_by_mean=t.index[-1],
                        n_holm_significant_pairs=int(pw["significant_holm_0.05"].sum()))
    return t


# ================================================================ Study 3B (Study 2 data)
e2 = pd.read_csv(S2 / "emotion_level_results.csv")
print(f"Study 2 emotion-level rows: {len(e2)} (seeds {sorted(e2.seed.unique())})")
report("Study3B, mBERT only (90 runs; marginal means over the other factor)", stability_per_seed(e2[e2.model == MBERT], "source", "target"), "Study3B_mBERT")
report("Study3B, all 4 models pooled (mean over models first; dominated by near-chance cells of monolingual models)",
       stability_per_seed(e2, "source", "target"), "Study3B_allmodels")

# ================================================================ Study 3A (Study 1 data)
e1 = pd.read_csv(S1 / "emotion_level_results.csv")
e1 = e1[e1.target.isin(["KO", "ZH", "EN"])].rename(columns={"source_language": "src"})
report("Study3A, Study 1 (targets KO/ZH/EN, both strategies pooled)", stability_per_seed(e1, "src", "target"), "Study3A_both_strategies")
report("Study3A, Study 1, mBERT only", stability_per_seed(e1[e1.strategy == "multilingual"], "src", "target"), "Study3A_mBERT")

# ================================================================ split-plot ANOVA with Emotion within run (mBERT, Study 2)
print("\n=== split-plot ANOVA, Study 2 mBERT: Source x Target between runs, Emotion within runs ===")
d = e2[e2.model == MBERT].copy()
d["run"] = d.seed.astype(str) + "_" + d.source + "_" + d.target
fx = smf.ols("f1 ~ C(source) * C(target) * C(emotion)", data=d).fit()
fa = sm.stats.anova_lm(fx, typ=2)
cellmean = d.groupby(["source", "target"]).f1.transform("mean")
run_means = d.groupby("run").f1.mean()
cell_of_run = d.assign(c=cellmean).groupby("run").c.mean()
n_e = len(EMO)
ss_runs = n_e * ((run_means - cell_of_run) ** 2).sum()
df_runs = d.run.nunique() - 9
ss_within_err = fa.loc["Residual", "sum_sq"] - ss_runs
df_within_err = fa.loc["Residual", "df"] - df_runs
rows = []
for term in fa.index[:-1]:
    between = "emotion" not in term
    es, ed = (ss_runs, df_runs) if between else (ss_within_err, df_within_err)
    ss, df = fa.loc[term, "sum_sq"], fa.loc[term, "df"]
    F = (ss / df) / (es / ed)
    rows.append(dict(term=term.replace("C(", "").replace(")", ""), stratum="between runs" if between else "within runs", df=df, df_error=ed,
                     F=F, p=stats.f.sf(F, df, ed), eta_sq=ss / fa.sum_sq.sum(), partial_eta_sq=ss / (ss + es)))
sp = pd.DataFrame(rows)
sp.to_csv(OUT / "Study3B_mBERT_splitplot_anova.csv", index=False)
print(sp.round(4).to_string(index=False))
summary["splitplot_mBERT"] = sp.set_index("term")[["F", "p", "eta_sq", "partial_eta_sq"]].to_dict()

# ================================================================ per-emotion F1 by path (for tables/figures)
path = e2[e2.model == MBERT].groupby(["source", "target", "emotion"]).f1.agg(["mean", "std"]).reset_index()
path.to_csv(OUT / "Study3B_mBERT_emotion_f1_by_path_mean_sd.csv", index=False)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(f"\nsaved to {OUT}")
