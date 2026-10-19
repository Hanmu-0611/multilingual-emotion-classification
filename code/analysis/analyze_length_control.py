#!/usr/bin/env python3
"""Sentence-level text-length control for Study 1.
Usage: python3 analyze_length_control.py <repo_root> <out_dir> <emotion_data_dir>
Unit = (target, strategy, sentence): mean correctness over the 10 seeds. Effect sizes only (sentences are shared across the 6 configs)."""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path
ROOT=Path(sys.argv[1]); OUT=Path(sys.argv[2]); DATA=Path(sys.argv[3]); OUT.mkdir(parents=True,exist_ok=True)  # DATA = your locally built emotion_data/ (contains text; not released)
S1=ROOT/"results/study1"
files={"KO":"dataset_experiment2_korean.csv","ZH":"dataset_experiment3_chinese.csv","EN":"dataset_experiment4_english.csv"}
q=pd.read_csv(S1/"prediction_level_results.csv.gz"); q=q[q.target.isin(files)]
q["hit"]=(q.true_label==q.predicted_label).astype(float)
g=q.groupby(["target","strategy","sample_id","source_language"]).hit.mean().reset_index()
rows=[]
for t,f in files.items():
    d=pd.read_csv(DATA/f).reset_index().rename(columns={"index":"sample_id"})
    d["len_input"]=d.text.astype(str).str.len()           # characters of the text the model saw
    d["len_orig"]=d.original_text.astype(str).str.len()   # characters of the untranslated sentence
    d["words_input"]=d.text.astype(str).str.split().str.len()
    x=g[g.target==t].merge(d[["sample_id","len_input","len_orig","words_input"]],on="sample_id"); rows.append(x)
d=pd.concat(rows,ignore_index=True)
print("units:",len(d))
print("\nmean text length by SourceLanguage (characters of the input the model saw / of the original sentence):")
print(d.drop_duplicates(["target","sample_id"]).groupby("source_language")[["len_input","len_orig"]].mean().round(1))
def pe(formula,label):
    m=smf.ols(formula,data=d).fit(); a=sm.stats.anova_lm(m,typ=2)
    ss=a.loc["C(source_language)","sum_sq"]; res=a.loc["Residual","sum_sq"]
    print(f"{label:55s} partial eta2(Source)={ss/(ss+res):.3f}  eta2={ss/a.sum_sq.sum():.3f}  R2={m.rsquared:.3f}")
    return ss/(ss+res), ss/a.sum_sq.sum(), m.rsquared
base="hit ~ C(source_language) + C(target)*C(strategy)"
r={}
r["baseline (no length)"]=pe(base,"baseline: Source + Target*Strategy")
r["+ char length of input"]=pe(base+" + len_input","+ character length of the input")
r["+ char length of original"]=pe(base+" + len_orig","+ character length of the original sentence")
r["+ log chars + squares"]=pe(base+" + np.log(len_input) + I(np.log(len_input)**2) + np.log(len_orig)","+ log length terms (input & original)")
r["+ whitespace words"]=pe(base+" + words_input","+ whitespace word count of the input")
pd.DataFrame(r,index=["partial_eta2_Source","eta2_Source","R2_model"]).T.round(4).to_csv(OUT/"length_control_study1.csv")
# stratified: accuracy by source within tertiles of ORIGINAL sentence length (computed over all sentences)
u=d.drop_duplicates(["target","sample_id"])[["sample_id","len_orig"]].drop_duplicates("sample_id")
d["len_bin"]=pd.qcut(d.len_orig,3,labels=["short","medium","long"],duplicates="drop")
print("\naccuracy by SourceLanguage within length terciles of the original sentence (mean over targets, strategies, seeds):")
tab=d.groupby(["len_bin","source_language"],observed=True).hit.mean().unstack().round(3); print(tab)
tab.to_csv(OUT/"length_strata_accuracy_by_source.csv")
print("\ncounts per cell:\n",d.groupby(["len_bin","source_language"],observed=True).sample_id.nunique().unstack())
