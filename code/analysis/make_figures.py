#!/usr/bin/env python3
"""Single-column (8.15 cm = 3.21 in) versions of the manuscript figures. Final printed font size >= 6.5 pt; markers differ for grayscale printing.
Column width = (210 - 21 - 21 - 5 mm gap) / 2 = 81.5 mm."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import sys
ROOT=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(parents=True,exist_ok=True)  # usage: python3 make_figures.py <repo_root> <out_dir>
W=3.21
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.8,"axes.titlesize":7,"axes.labelsize":6.8,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.3,"axes.linewidth":.6,"lines.linewidth":1.0})
RNG=np.random.default_rng(0); L=["KO","ZH","EN"]; EMO=["anger","fear","joy","sadness","surprise"]
# ---------------- Fig 1
fig,ax=plt.subplots(figsize=(W,4.35)); ax.set_xlim(0,100); ax.set_ylim(-4,100); ax.axis("off")
def box(x,y,w,h,t,fc,ec,fs=6.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.5",fc=fc,ec=ec,lw=.8)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,linespacing=1.2)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6,lw=.8,color="#333",shrinkA=0,shrinkB=0))
Y,YE,B,BE,G,GE="#F5EBD7","#8A6D2F","#E8EEF7","#2F4B7C","#E7F2E8","#3B7A57"
box(5,84,90,13,"Public datasets (five emotions):\nKorEmpatheticDialogues (KO), CLUEEmotion-\nAnalysis2020 (ZH), GoEmotions (EN)",Y,YE)
box(5,64,90,13,"9,000 sentences (3 sources × 5 emotions × 600)\nfixed train / test split",Y,YE)
box(5,44,90,13,"GPT-4o-mini translation: every sentence in\nKO / ZH / EN versions",B,BE)
box(1,17,31,19,"Study 1\norigin dataset ×\ninput language ×\nmodel strategy\n60 runs",B,BE,fs=6.0)
box(35,17,31,19,"Study 2\n9 train→test\nlanguage paths\n× 4 models\n360 runs",B,BE,fs=6.0)
box(69,17,30,19,"Study 3\nemotion-wise\nstability,\nsadness errors",B,BE,fs=6.0)
box(5,0,90,12,"Supplementary: translation check, NLLB re-run,\nlength control, protocol effects",G,GE,fs=6.0)
arr(50,84,50,77.6); arr(50,64,50,57.6); arr(50,44,16,36.6); arr(50,44,50,36.6); arr(50,44,84,36.6)
fig.savefig(OUT/"Fig1_workflow.png",dpi=400,bbox_inches="tight",pad_inches=0.03); plt.close(fig)
# ---------------- Fig 2
d=pd.read_csv(ROOT/"results/analysis/study1/table_source_level_cell_means.csv")
fig,axes=plt.subplots(2,1,figsize=(W,3.7),sharex=True,sharey=True)
for ax,s,ttl in zip(axes,["multilingual","language_specific"],["(a) Multilingual strategy (mBERT)","(b) Language-specific strategy"]):
    for t,c,mk,ls in zip(L,["#1f77b4","#d95f02","#2ca02c"],["o","s","^"],["-","--",":"]):
        x=d[(d.strategy==s)&(d.target==t)].set_index("source_language").loc[["ko","zh","en"]]
        ax.errorbar(["KO","ZH","EN"],x["mean"],yerr=[x["mean"]-x.ci_lo,x.ci_hi-x["mean"]],marker=mk,ms=3.2,capsize=1.5,ls=ls,color=c,label=f"Target {t}")
    ax.set_title(ttl,loc="left"); ax.grid(alpha=.3,lw=.4); ax.set_ylabel("Macro-F1")
axes[1].set_xlabel("SourceLanguage (origin dataset of test sentences)"); axes[0].legend(loc="upper right",ncol=3,handlelength=2.2,columnspacing=.8)
fig.tight_layout(pad=.4); fig.savefig(OUT/"Fig2_study1_profile.png",dpi=400); plt.close(fig)
# ---------------- Fig 3
t=pd.read_csv(ROOT/"results/analysis/study2/table_cell_means_sd_ci.csv")
names={"bert-base-multilingual-cased":"mBERT","klue/roberta-base":"KLUE-RoBERTa","hfl/chinese-roberta-wwm-ext":"Chinese-RoBERTa","roberta-base":"RoBERTa-base"}
fig,axes=plt.subplots(2,2,figsize=(W,3.25),constrained_layout=True)
for ax,(m,n) in zip(axes.ravel(),names.items()):
    M=np.array([[t[(t.model==m)&(t.source==s)&(t.target==g)]["mean"].iloc[0] for g in L] for s in L])
    im=ax.imshow(M,vmin=0,vmax=0.7,cmap="viridis")
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=6.3,color="white" if M[i,j]<0.4 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(L); ax.set_yticks(range(3)); ax.set_yticklabels(L); ax.set_title(n,fontsize=6.8)
for ax in axes[1]: ax.set_xlabel("Test language")
for ax in axes[:,0]: ax.set_ylabel("Training language")
fig.colorbar(im,ax=axes,shrink=.8,label="Macro-F1",pad=.03); fig.savefig(OUT/"Fig3_study2_heatmaps.png",dpi=400); plt.close(fig)
# ---------------- Fig 4 (translator robustness, filename kept)
c=pd.read_csv(ROOT/"results/study2_nllb_mbert/compare_GPT_vs_NLLB_mBERT_cells.csv"); same=c.source==c.target
fig,ax=plt.subplots(figsize=(W,2.9))
ax.scatter(c.gpt[~same],c.nllb[~same],c="#d95f02",marker="s",s=14,label="cross-language")
ax.scatter(c.gpt[same],c.nllb[same],c="#1f77b4",marker="o",s=16,label="same-language")
for _,r in c.iterrows(): ax.annotate(f"{r.source}→{r.target}",(r.gpt,r.nllb),fontsize=5.6,xytext=(2,2),textcoords="offset points")
lo,hi=0.33,0.62; ax.plot([lo,hi],[lo,hi],"k--",lw=.6); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_xlabel("Macro-F1, GPT-4o-mini translations"); ax.set_ylabel("Macro-F1, NLLB-200 translations"); ax.legend(loc="lower right"); ax.grid(alpha=.3,lw=.4)
fig.tight_layout(pad=.4); fig.savefig(OUT/"Fig4_translator_robustness.png",dpi=400); plt.close(fig)
# ---------------- Fig 5 (emotion stability, filename kept)
def ci(x): m=RNG.choice(x,(10000,len(x))).mean(1); return np.percentile(m,[2.5,97.5])
fig,axes=plt.subplots(2,1,figsize=(W,3.9))
for ax,key,ttl in zip(axes,["Study3A_both_strategies","Study3B_mBERT"],["(a) Study 1 (both strategies)","(b) Study 2 (mBERT)"]):
    p=pd.read_csv(ROOT/f"results/analysis/study3/{key}_stability_per_seed.csv")
    mu=[p[p.emotion==e].sd.mean() for e in EMO]; cis=[ci(p[p.emotion==e].sd.values) for e in EMO]
    ax.bar(EMO,mu,yerr=[[m-q[0] for m,q in zip(mu,cis)],[q[1]-m for m,q in zip(mu,cis)]],capsize=2,color="#8FA9D0",edgecolor="#2F4B7C",lw=.6,error_kw=dict(lw=.7))
    for i,e in enumerate(EMO): ax.scatter([i]*10,p[p.emotion==e].sd,s=3.5,c="k",alpha=.5,zorder=3,lw=0)
    ax.set_title(ttl,loc="left"); ax.set_ylabel("Cross-condition SD of F1"); ax.grid(axis="y",alpha=.3,lw=.4)
fig.tight_layout(pad=.4); fig.savefig(OUT/"Fig5_emotion_stability.png",dpi=400); plt.close(fig)
print("done")
