#!/usr/bin/env python3
"""
Study 1 re-run with the final-epoch protocol and real replicates (reviewer 2 #3, #4, #5, #7).

What changed relative to the submitted Study 1 script (研究1_训练脚本.py)
  * No checkpoint selection on the test set: load_best_model_at_end=False, the model after epoch 3 is evaluated once.
  * 10 training seeds per configuration (the data files and the train/test split are the same for every seed;
    only head initialisation, dropout and batch order change). Seeds are real replicates for a run-level analysis.
  * Model Strategy is only defined where it exists. The Original (mixed-language) dataset has no language-specific
    model, so it is run ONCE with mBERT and reported as a reference condition (not as a Model Strategy level).
    Strategy comparison: TargetLanguage in {KO, ZH, EN} x Model {mBERT (multilingual), language-specific}.
  * Saves run-level, per-source-language, emotion-level and prediction-level (with sample_id) results.
Everything else (data files, max_length=128, lr 2e-5, batch 16, 3 epochs, weight decay 0.01, models) is unchanged.

Inputs : emotion_data/dataset_experiment{1,2,3,4}_*.csv  (columns: text, original_text, original_language, emotion, label, split)
Outputs: outputs/study1_fixed_10seeds/{run_level,source_level,emotion_level,prediction_level}_results.csv,
         failed_runs.csv, environment_report.txt, completeness_check.txt
Resumable: re-running skips runs already present in run_level_results.csv.
Env vars: SEEDS (comma list, default all 10), SMOKE=1 (tiny subsample, for a quick dry run only)
"""
from __future__ import annotations

import csv
import gc
import inspect
import os
import platform
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 2026, 7, 17, 29, 53, 71, 101, 31415]
SEEDS = [int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else DEFAULT_SEEDS
SMOKE = os.environ.get("SMOKE") == "1"

DATA_DIR = Path("emotion_data")
OUTPUT_DIR = Path("outputs/study1_fixed_10seeds" + ("_smoke" if SMOKE else ""))
RUNS_DIR = OUTPUT_DIR / "model_runs"
RUN_CSV = OUTPUT_DIR / "run_level_results.csv"
SOURCE_CSV = OUTPUT_DIR / "source_level_results.csv"
EMOTION_CSV = OUTPUT_DIR / "emotion_level_results.csv"
PRED_CSV = OUTPUT_DIR / "prediction_level_results.csv"
FAILED_CSV = OUTPUT_DIR / "failed_runs.csv"
ENV_TXT = OUTPUT_DIR / "environment_report.txt"
COMPLETE_TXT = OUTPUT_DIR / "completeness_check.txt"

EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]
SOURCES = ["ko", "zh", "en"]
MBERT = "bert-base-multilingual-cased"

# (target, dataset file, model strategy, model name)
CONFIGS = [
    ("Original", "dataset_experiment1_original.csv", "multilingual", MBERT),  # reference condition, run once
    ("KO", "dataset_experiment2_korean.csv", "multilingual", MBERT),
    ("KO", "dataset_experiment2_korean.csv", "language_specific", "klue/roberta-base"),
    ("ZH", "dataset_experiment3_chinese.csv", "multilingual", MBERT),
    ("ZH", "dataset_experiment3_chinese.csv", "language_specific", "hfl/chinese-roberta-wwm-ext"),
    ("EN", "dataset_experiment4_english.csv", "multilingual", MBERT),
    ("EN", "dataset_experiment4_english.csv", "language_specific", "roberta-base"),
]


@dataclass(frozen=True)
class Run:
    seed: int
    target: str
    file: str
    strategy: str
    model: str

    @property
    def key(self):
        return (self.seed, self.target, self.strategy, self.model)

    @property
    def tmp_dir(self) -> Path:
        return RUNS_DIR / f"seed{self.seed}_{self.target}_{self.strategy}"


def all_runs():
    return [Run(s, *c) for s in SEEDS for c in CONFIGS]


def section(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)


def append_rows(path: Path, rows, fields):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


RUN_FIELDS = ["seed", "target", "strategy", "model", "accuracy", "macro_f1", "f1_epoch1", "f1_epoch2", "f1_epoch3",
              "f1_best_of_3", "n_train", "n_test", "train_runtime_sec", "total_runtime_sec", "torch_version",
              "transformers_version", "gpu_name", "timestamp"]
SOURCE_FIELDS = ["seed", "target", "strategy", "model", "source_language", "accuracy", "macro_f1", "n_test"]
EMOTION_FIELDS = ["seed", "target", "strategy", "model", "source_language", "emotion", "precision", "recall", "f1", "support"]
PRED_FIELDS = ["seed", "target", "strategy", "model", "sample_id", "source_language", "true_label", "predicted_label"] + \
              [f"prob_{e}" for e in EMOTIONS]
FAILED_FIELDS = ["seed", "target", "strategy", "model", "error_type", "error_message", "timestamp"]


def completed_keys():
    if not RUN_CSV.exists():
        return set()
    import pandas as pd
    d = pd.read_csv(RUN_CSV)
    return set(zip(d["seed"], d["target"], d["strategy"], d["model"]))


def load_data(run: Run, pd):
    df = pd.read_csv(DATA_DIR / run.file)
    need = {"text", "original_text", "original_language", "emotion", "split"}
    assert need <= set(df.columns), f"{run.file} missing {need - set(df.columns)}"
    df = df.reset_index().rename(columns={"index": "sample_id"})  # row position in the shipped file = stable id
    df["text"] = df["text"].astype(str)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    assert set(df["emotion"]) <= set(EMOTIONS), set(df["emotion"])
    train, test = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    if SMOKE:
        train = train.groupby("emotion", group_keys=False).head(8).reset_index(drop=True)
        test = test.groupby(["original_language", "emotion"], group_keys=False).head(2).reset_index(drop=True)
    # leakage guard: no test sentence may also be a training sentence
    overlap = float(test["original_text"].isin(set(train["original_text"])).mean())
    assert overlap < 0.02, f"train/test overlap {overlap:.4f}"
    return train, test


def run_one(run: Run, rt):
    pd, np, torch = rt["pd"], rt["np"], rt["torch"]
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    t0 = time.time()
    section(f"seed={run.seed} | target={run.target} | {run.strategy} | {run.model}")
    train, test = load_data(run, pd)
    l2i = {e: i for i, e in enumerate(EMOTIONS)}
    train["label"], test["label"] = train["emotion"].map(l2i), test["emotion"].map(l2i)

    random.seed(run.seed)
    np.random.seed(run.seed)
    rt["set_seed"](run.seed)
    tok = rt["AutoTokenizer"].from_pretrained(run.model)
    enc = lambda b: tok(b["text"], truncation=True, max_length=128)
    ds_tr = rt["Dataset"].from_pandas(train[["text", "label"]], preserve_index=False).map(enc, batched=True)
    ds_te = rt["Dataset"].from_pandas(test[["text", "label"]], preserve_index=False).map(enc, batched=True)
    model = rt["AutoModelForSequenceClassification"].from_pretrained(
        run.model, num_labels=5, id2label=dict(enumerate(EMOTIONS)), label2id=l2i, ignore_mismatched_sizes=True)

    def metrics(p):
        pr = np.argmax(p.predictions, axis=-1)
        return {"Accuracy": accuracy_score(p.label_ids, pr), "Macro_F1": f1_score(p.label_ids, pr, average="macro", zero_division=0)}

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(output_dir=str(run.tmp_dir), seed=run.seed, data_seed=run.seed, num_train_epochs=3, learning_rate=2e-5,
              per_device_train_batch_size=16, per_device_eval_batch_size=16, weight_decay=0.01, save_strategy="no",
              logging_steps=200, load_best_model_at_end=False, report_to=[], disable_tqdm=True)
    kw["evaluation_strategy" if "evaluation_strategy" in inspect.signature(rt["TrainingArguments"].__init__).parameters else "eval_strategy"] = "epoch"
    targs = rt["TrainingArguments"](**kw)
    tk = dict(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_te,
              data_collator=rt["DataCollatorWithPadding"](tokenizer=tok), compute_metrics=metrics)
    tk["tokenizer" if "tokenizer" in inspect.signature(rt["Trainer"].__init__).parameters else "processing_class"] = tok
    trainer = rt["Trainer"](**tk)

    t1 = time.time()
    trainer.train()
    train_rt = time.time() - t1
    out = trainer.predict(ds_te)  # final-epoch model, evaluated once
    logits, y = out.predictions, out.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = np.argmax(logits, axis=-1)
    ep = [float(e["eval_Macro_F1"]) for e in trainer.state.log_history if "eval_Macro_F1" in e]
    assert len(ep) == 3, ep
    macro = float(f1_score(y, pred, average="macro", zero_division=0))
    assert abs(macro - ep[2]) < 1e-9, (macro, ep)  # last-epoch eval == final model

    base = dict(seed=run.seed, target=run.target, strategy=run.strategy, model=run.model)
    src = test["original_language"].values
    src_rows, emo_rows = [], []
    for s in SOURCES:
        m = src == s
        src_rows.append({**base, "source_language": s, "accuracy": float(accuracy_score(y[m], pred[m])),
                         "macro_f1": float(f1_score(y[m], pred[m], average="macro", labels=list(range(5)), zero_division=0)),
                         "n_test": int(m.sum())})
        p, r, f, sup = precision_recall_fscore_support(y[m], pred[m], labels=list(range(5)), zero_division=0)
        for i, e in enumerate(EMOTIONS):
            emo_rows.append({**base, "source_language": s, "emotion": e, "precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(sup[i])})
    pred_rows = []
    for i in range(len(y)):
        pred_rows.append({**base, "sample_id": int(test["sample_id"][i]), "source_language": src[i],
                          "true_label": EMOTIONS[int(y[i])], "predicted_label": EMOTIONS[int(pred[i])],
                          **{f"prob_{e}": float(probs[i][j]) for j, e in enumerate(EMOTIONS)}})
    append_rows(SOURCE_CSV, src_rows, SOURCE_FIELDS)
    append_rows(EMOTION_CSV, emo_rows, EMOTION_FIELDS)
    append_rows(PRED_CSV, pred_rows, PRED_FIELDS)
    # run-level row is written LAST: it is the completion marker used for resuming
    append_rows(RUN_CSV, [{**base, "accuracy": float(accuracy_score(y, pred)), "macro_f1": macro,
                           "f1_epoch1": ep[0], "f1_epoch2": ep[1], "f1_epoch3": ep[2], "f1_best_of_3": max(ep),
                           "n_train": len(train), "n_test": len(test), "train_runtime_sec": round(train_rt, 1),
                           "total_runtime_sec": round(time.time() - t0, 1), "torch_version": torch.__version__,
                           "transformers_version": rt["transformers"].__version__,
                           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}], RUN_FIELDS)
    print(f"Done in {time.time() - t0:.0f}s | Macro_F1={macro:.4f}", flush=True)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(run.tmp_dir, ignore_errors=True)


def purge_partial(run: Run):
    """If a previous attempt died after writing some tier but before the run-level marker, drop those rows."""
    import pandas as pd
    for path in (SOURCE_CSV, EMOTION_CSV, PRED_CSV):
        if path.exists():
            d = pd.read_csv(path)
            keep = ~((d.seed == run.seed) & (d.target == run.target) & (d.strategy == run.strategy) & (d.model == run.model))
            if not keep.all():
                d[keep].to_csv(path, index=False)


def completeness():
    import pandas as pd
    exp = {r.key for r in all_runs()}
    d = pd.read_csv(RUN_CSV) if RUN_CSV.exists() else pd.DataFrame(columns=["seed", "target", "strategy", "model"])
    act = list(zip(d["seed"], d["target"], d["strategy"], d["model"]))
    cnt = Counter(act)
    miss = sorted(exp - set(act))
    dup = [k for k, v in cnt.items() if v > 1]
    lines = ["Study 1 fixed re-run -- completeness check", "=" * 70,
             f"Expected {len(SEEDS)} seeds x {len(CONFIGS)} configs = {len(exp)} runs; present {len(set(act))}; missing {len(miss)}; duplicated {len(dup)}"]
    lines += [f"  MISSING {m}" for m in miss] + [f"  DUPLICATE {k}" for k in dup]
    if RUN_CSV.exists():
        for name, path, per in (("source_level", SOURCE_CSV, 3), ("emotion_level", EMOTION_CSV, 15), ("prediction_level", PRED_CSV, 1500)):
            n = len(pd.read_csv(path)) if path.exists() else 0
            lines.append(f"{name}: {n} rows (expected {len(set(act)) * per if not SMOKE else 'n/a (smoke)'})")
    COMPLETE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    need = [DATA_DIR / c[1] for c in CONFIGS]
    miss = [p for p in need if not p.exists()]
    if miss:
        sys.exit(f"Missing data files: {miss}")
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)
    rt = dict(np=np, pd=pd, torch=torch, transformers=transformers, Dataset=Dataset, set_seed=set_seed,
              AutoModelForSequenceClassification=AutoModelForSequenceClassification, AutoTokenizer=AutoTokenizer,
              DataCollatorWithPadding=DataCollatorWithPadding, Trainer=Trainer, TrainingArguments=TrainingArguments)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ENV_TXT.write_text("\n".join([
        "Study 1 fixed re-run -- environment", f"python {sys.version}", f"os {platform.platform()}",
        f"torch {torch.__version__}  cuda {torch.cuda.is_available()}",
        f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}",
        f"transformers {transformers.__version__}", f"seeds {SEEDS}", f"configs {CONFIGS}",
        f"started {time.strftime('%Y-%m-%d %H:%M:%S')}"]) + "\n", encoding="utf-8")
    done = completed_keys()
    runs = all_runs()
    section(f"Runs: {len(runs)} (already done: {len(done & {r.key for r in runs})})")
    for r in runs:
        if r.key in done:
            print(f"Skip completed: {r.key}")
            continue
        try:
            purge_partial(r)
            run_one(r, rt)
        except Exception as exc:  # noqa: BLE001 -- failures must be recorded, never silently skipped
            print(f"FAILED: {r.key} -> {exc}", file=sys.stderr)
            traceback.print_exc()
            append_rows(FAILED_CSV, [{"seed": r.seed, "target": r.target, "strategy": r.strategy, "model": r.model,
                                      "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}], FAILED_FIELDS)
            shutil.rmtree(r.tmp_dir, ignore_errors=True)
    completeness()


if __name__ == "__main__":
    main()
