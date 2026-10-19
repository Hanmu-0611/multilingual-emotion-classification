#!/usr/bin/env python3
"""Extra analyses (all from the released results; no new training).

Usage: python3 analyze_extras.py <repo_root> <out_dir>
(The comparison with the earlier, flawed evaluation design is not included because that design's results are not released.)

A1 effect of picking the best epoch on the test set = f1_best_of_3 - f1_last (clean data, same runs)
B. R3 #5  Sadness: where is it misclassified? (confusion matrices / predictions) + is its 'sensitivity' driven by recall or precision?
C. R2 #8  Test-sentence bootstrap CIs (uncertainty from the finite, fixed test sample, which seeds cannot capture)
D.        Independent cross-check of the hand-built Study 1 split-plot ANOVA
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
S2 = ROOT / "results/study2"
S1 = ROOT / "results/study1"
RNG = np.random.default_rng(0)
EMO = ["anger", "fear", "joy", "sadness", "surprise"]
MBERT = "bert-base-multilingual-cased"
pd.set_option("display.width", 220)
summary = {}


def boot(x, n=10000):
    x = np.asarray(x, float)
    m = RNG.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


# ============================================================================ A1
print("=" * 90 + "\nA1. Effect of picking the best epoch on the TEST set (clean runs: best-of-3 minus last epoch)\n" + "=" * 90)
r2 = pd.read_csv(S2 / "run_level_results.csv")
r1 = pd.read_csv(S1 / "run_level_results.csv")
r2["infl"] = r2.f1_best_of_3 - r2.f1_last
r1["infl"] = r1.f1_best_of_3 - r1.macro_f1
for name, d in [("Study 2 (360 runs)", r2), ("Study 1 (70 runs)", r1)]:
    lo, hi = boot(d.infl)
    print(f"{name}: mean inflation = {d.infl.mean():+.4f} F1 (95% CI {lo:.4f} to {hi:.4f}); "
          f"max {d.infl.max():.3f}; runs where the best epoch is not the last: {(d.infl > 1e-12).mean() * 100:.1f}%")
    summary[f"A1_{name.split()[0]}{name.split()[1]}"] = dict(mean=float(d.infl.mean()), ci=[float(lo), float(hi)], share_best_not_last=float((d.infl > 1e-12).mean()))
a1 = r2.groupby("model").infl.agg(["mean", "max"]).round(4)
a1["share_runs_best_not_last"] = r2.groupby("model").infl.apply(lambda x: (x > 1e-12).mean()).round(3)
print("\nStudy 2 by model:\n", a1.to_string())
a1.to_csv(OUT / "A1_checkpoint_selection_inflation_by_model_study2.csv")
a1c = r2.groupby(["source", "target", "model"]).infl.mean().reset_index()
a1c.to_csv(OUT / "A1_checkpoint_selection_inflation_by_cell_study2.csv", index=False)
r1.groupby(["target", "strategy"]).infl.agg(["mean", "max"]).round(4).to_csv(OUT / "A1_checkpoint_selection_inflation_study1.csv")

# The reported numbers come from a script that also compared the results with an earlier, flawed evaluation design;
# that comparison used two bootstrap draws here. Re-draw them so the random stream - and hence every bootstrap
# interval below - is identical to the reported values.
for _ in range(2):
    RNG.choice(np.zeros(9), (10000, 9))

# ============================================================================ B
print("\n" + "=" * 90 + "\nB. Sadness: confusion structure and the source of its variability\n" + "=" * 90)
cm_files = sorted(glob.glob(str(S2 / "confusion_matrices/*.csv")))
mats = {}
for f in cm_files:
    nm = Path(f).stem  # seed42_ko_to_ko_bert-base-multilingual-cased
    if MBERT not in nm:
        continue
    seed, pair = nm.split("_")[0], "_".join(nm.split("_")[1:4])
    mats[(int(seed.replace("seed", "")), pair)] = pd.read_csv(f, index_col=0).loc[EMO, EMO].values.astype(float)
pairs = sorted({p for _, p in mats})
assert len(mats) == 90, len(mats)
rows = []
for pr in pairs:
    M = np.mean([mats[(s, pr)] for s in sorted({s for s, p in mats if p == pr})], axis=0)  # mean counts over seeds
    rec = M / M.sum(1, keepdims=True)
    prec = M / M.sum(0, keepdims=True)
    si = EMO.index("sadness")
    off = {e: rec[si, EMO.index(e)] for e in EMO if e != "sadness"}
    into = {e: prec[EMO.index(e), si] if False else M[EMO.index(e), si] / M[:, si].sum() for e in EMO if e != "sadness"}
    top = max(off, key=off.get)
    rows.append(dict(pair=pr.replace("_to_", "->").upper(), same_language=pr.split("_")[0] == pr.split("_")[2],
                     sadness_recall=rec[si, si], sadness_precision=prec[si, si],
                     **{f"sad_to_{e}": off[e] for e in off}, top_confusion_from_sadness=top,
                     **{f"mean_recall_{e}": rec[EMO.index(e), EMO.index(e)] for e in EMO}))
b1 = pd.DataFrame(rows)
b1.to_csv(OUT / "B1_sadness_confusion_by_path_mBERT_study2.csv", index=False)
print("B1. Study 2, mBERT, mean confusion over 10 seeds (row-normalised). Sadness row:")
show = b1[["pair", "sadness_recall", "sadness_precision", "sad_to_anger", "sad_to_fear", "sad_to_joy", "sad_to_surprise", "top_confusion_from_sadness"]]
print(show.round(3).to_string(index=False))
rec_cols = [f"mean_recall_{e}" for e in EMO]
print("\nmean recall over the 9 paths:", b1[rec_cols].mean().round(3).rename(lambda c: c.replace("mean_recall_", "")).to_dict())
print("mean recall same-language paths:", b1[b1.same_language][rec_cols].mean().round(3).rename(lambda c: c.replace("mean_recall_", "")).to_dict())
print("mean recall cross-language paths:", b1[~b1.same_language][rec_cols].mean().round(3).rename(lambda c: c.replace("mean_recall_", "")).to_dict())
print("share of sadness errors going to each emotion, averaged over the 9 paths:", {e: round(b1[f'sad_to_{e}'].mean() / sum(b1[f'sad_to_{x}'].mean() for x in EMO if x != 'sadness'), 3) for e in EMO if e != "sadness"})

# B2 recall vs precision variability (6 marginal means, per seed) — Study 2 mBERT and Study 1
def stab(df, src, tgt, col):
    out = {}
    for seed, g in df.groupby("seed"):
        for e in EMO:
            p = g[g.emotion == e]
            v = np.concatenate([p.groupby(src)[col].mean().values, p.groupby(tgt)[col].mean().values])
            out.setdefault(e, []).append(v.std(ddof=1))
    return {e: np.mean(v) for e, v in out.items()}
e2 = pd.read_csv(S2 / "emotion_level_results.csv"); e2 = e2[e2.model == MBERT]
e1 = pd.read_csv(S1 / "emotion_level_results.csv"); e1 = e1[e1.target.isin(["KO", "ZH", "EN"])].rename(columns={"source_language": "src"})
tab = []
for lab, df, s_, t_ in [("Study 2 (mBERT)", e2, "source", "target"), ("Study 1", e1, "src", "target")]:
    for col in ["precision", "recall", "f1"]:
        d = stab(df, s_, t_, col)
        tab.append(pd.Series(d, name=f"{lab} | {col}"))
b2 = pd.DataFrame(tab)[EMO].round(3)
b2.to_csv(OUT / "B2_stability_sd_precision_recall_f1.csv")
print("\nB2. Cross-condition SD (6 marginal means, mean over seeds) of precision / recall / F1 per emotion:\n", b2.to_string())

# B3 Study 1 predictions: sadness by source language
q1 = pd.read_csv(S1 / "prediction_level_results.csv.gz")
q1 = q1[q1.target.isin(["KO", "ZH", "EN"])]
rows = []
for (src, strat, tgt), g in q1.groupby(["source_language", "strategy", "target"]):
    s = g[g.true_label == "sadness"]
    rows.append(dict(source_language=src, strategy=strat, target=tgt, sadness_recall=(s.predicted_label == "sadness").mean(),
                     **{f"sad_to_{e}": (s.predicted_label == e).mean() for e in EMO if e != "sadness"}))
b3 = pd.DataFrame(rows)
b3.to_csv(OUT / "B3_study1_sadness_confusion.csv", index=False)
agg = b3.groupby("source_language")[["sadness_recall"] + [f"sad_to_{e}" for e in EMO if e != "sadness"]].mean().round(3)
print("\nB3. Study 1, sadness row of the confusion matrix by source language (mean over targets x strategies, 10 seeds):\n", agg.to_string())
allrec = q1.assign(hit=lambda d: d.true_label == d.predicted_label).groupby(["source_language", "true_label"]).hit.mean().unstack().round(3)[EMO]
print("\nStudy 1 recall of every emotion by source language:\n", allrec.to_string())
allrec.to_csv(OUT / "B3_study1_recall_by_source_and_emotion.csv")

# ============================================================================ C
print("\n" + "=" * 90 + "\nC. Test-sentence bootstrap (1000 resamples of the fixed test sentences; mean over the 10 seeds)\n" + "=" * 90)
q2 = pd.read_csv(S2 / "prediction_level_results.csv.gz")
q2 = q2[q2.model == MBERT]
lab = {e: i for i, e in enumerate(EMO)}


def boot_macro_f1(trues, preds, B=1000):
    """trues/preds: list over seeds of (n,) int arrays on the SAME sentences; returns B bootstrap values of the seed-averaged macro-F1."""
    n = len(trues[0])
    W = RNG.multinomial(n, np.ones(n) / n, size=B).astype(float)
    res = np.zeros(B)
    for t, p in zip(trues, preds):
        T = np.eye(5)[t]; P = np.eye(5)[p]
        tp, ts, ps = W @ (T * P), W @ T, W @ P
        res += np.where(ts + ps > 0, 2 * tp / np.maximum(ts + ps, 1e-12), 0).mean(1)
    return res / len(trues)


rows = []
store = {}
for (s, t), g in q2.groupby(["source", "target"]):
    seeds_ = sorted(g.seed.unique())
    ids0 = None; tr = []; pr = []
    for sd in seeds_:
        x = g[g.seed == sd].sort_values("sample_id")
        if ids0 is None: ids0 = x.sample_id.values
        assert (x.sample_id.values == ids0).all()
        tr.append(x.true_label.map(lab).values); pr.append(x.predicted_label.map(lab).values)
    bs = boot_macro_f1(tr, pr)
    store[(s, t)] = bs
    point = np.mean([np.mean([2 * ((tr[i] == k) & (pr[i] == k)).sum() / max(((tr[i] == k).sum() + (pr[i] == k).sum()), 1) for k in range(5)]) for i in range(len(tr))])
    seed_f1 = [np.mean([2 * ((tr[i] == k) & (pr[i] == k)).sum() / max(((tr[i] == k).sum() + (pr[i] == k).sum()), 1) for k in range(5)]) for i in range(len(tr))]
    rows.append(dict(source=s, target=t, mean_f1=point, test_boot_ci_lo=np.percentile(bs, 2.5), test_boot_ci_hi=np.percentile(bs, 97.5),
                     seed_ci_lo=boot(seed_f1)[0], seed_ci_hi=boot(seed_f1)[1]))
c = pd.DataFrame(rows)
c["test_ci_halfwidth"] = (c.test_boot_ci_hi - c.test_boot_ci_lo) / 2
c["seed_ci_halfwidth"] = (c.seed_ci_hi - c.seed_ci_lo) / 2
c.to_csv(OUT / "C_study2_mBERT_test_sentence_bootstrap_ci.csv", index=False)
print(c.round(3).to_string(index=False))
print(f"\nmean CI half-width: test-sentence bootstrap {c.test_ci_halfwidth.mean():.3f} vs seed bootstrap {c.seed_ci_halfwidth.mean():.3f} "
      f"(sentence sampling adds {c.test_ci_halfwidth.mean() / c.seed_ci_halfwidth.mean():.1f}x the uncertainty of seed variation alone)")
# does the diagonal advantage survive test-sentence resampling?
diag = np.mean([store[(k, k)] for k in ["KO", "ZH", "EN"]], axis=0)
off = np.mean([store[(a, b)] for a in ["KO", "ZH", "EN"] for b in ["KO", "ZH", "EN"] if a != b], axis=0)
gap = diag - off
print(f"same-language minus cross-language F1 (mBERT): {gap.mean():.3f}, 95% test-bootstrap CI {np.percentile(gap, 2.5):.3f} to {np.percentile(gap, 97.5):.3f}")
summary["C_gap_same_minus_cross"] = dict(mean=float(gap.mean()), ci=[float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))])

# ============================================================================ D
print("\n" + "=" * 90 + "\nD. Independent cross-check of the Study 1 split-plot ANOVA\n" + "=" * 90)
s1 = pd.read_csv(S1 / "source_level_results.csv")
s1 = s1[s1.target.isin(["KO", "ZH", "EN"])].copy()
s1["run"] = s1.seed.astype(str) + "_" + s1.target + "_" + s1.strategy
rm = s1.groupby(["run", "target", "strategy"]).macro_f1.mean().reset_index()
fb = sm.stats.anova_lm(smf.ols("macro_f1 ~ C(target)*C(strategy)", data=rm).fit(), typ=2)
print("between-run part = ANOVA on run means (60 runs, df_error=54):\n", fb[["df", "F"]].round(3).to_string())
sp = pd.read_csv(Path("分析结果_Study1_修复版/anova_split_plot_target_model_source.csv")) if Path("分析结果_Study1_修复版/anova_split_plot_target_model_source.csv").exists() else None
# within-run part: remove run means, then F against the Source x run error
s1["y_c"] = s1.macro_f1 - s1.groupby("run").macro_f1.transform("mean")
fw = smf.ols("y_c ~ C(target)*C(strategy)*C(source_language)", data=s1).fit()
aw = sm.stats.anova_lm(fw, typ=2)
df_err = 108
ss_err = aw.loc["Residual", "sum_sq"]
within = {}
for term in aw.index[:-1]:
    if "source_language" in term:
        F = (aw.loc[term, "sum_sq"] / aw.loc[term, "df"]) / (ss_err / df_err)
        within[term] = round(float(F), 3)
print("within-run part from run-centred data (df_error=108):", within)
summary["D_between_F"] = {k: round(float(v), 3) for k, v in fb["F"].dropna().items()}
summary["D_within_F"] = within
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("\nsaved to", OUT)
