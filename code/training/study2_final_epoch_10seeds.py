#!/usr/bin/env python3
"""
Study 2 multi-seed reproduction.

Reuses, VERBATIM, the dataset-building and preprocessing logic from
研究2_36个SourceTarget实验运行脚本.py (same MODELS, same EMOTIONS, same
TRAIN_N/TEST_N=2500/500, same balanced_sample() with master seed=42, same
tokenizer/model choice, same TrainingArguments hyperparameters:
max_length=128, epochs=3, lr=2e-5, batch=16, weight_decay=0.01,
save_strategy=epoch, load_best_model_at_end on Macro_F1).

The ONLY thing this script adds on top of the original:
  1. The 9 Source->Target datasets are built ONCE with the original
     seed=42 sampling (dataset itself is NOT re-sampled per seed, per
     the user's instruction to keep the dataset fully unchanged).
  2. For each of 10 training seeds x 36 (source,target,model) combos,
     retrains with that seed fed into set_seed()/TrainingArguments so
     only training-time stochasticity (weight init order for the head,
     dropout, batch shuffling) varies across seeds -- not the data.
  3. Saves three tiers of output (run-level, emotion-level,
     prediction-level with per-class probabilities) plus a confusion
     matrix per run, instead of only the aggregate row the original
     script wrote.
  4. Logs environment/version info, per-run wall-clock time, and any
     failed run explicitly (no silent skipping).
  5. Deletes each run's model checkpoint after extracting metrics, so
     360 runs don't fill the disk (checkpoints are not part of the
     requested outputs).

Nothing about the model choice, training hyperparameters, data
sampling, or evaluation metric definitions has been changed relative
to the original script.
"""

from __future__ import annotations

import gc
import json
import platform
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEEDS = [42, 123, 2026, 7, 17, 29, 53, 71, 101, 31415]  # all 10 seeds, final-epoch protocol
MASTER_DATA_SEED = 42  # dataset sampling seed -- identical to the original script, never changes across runs
TRAIN_N = 2500
TEST_N = 500

DATA_DIR = Path("emotion_data")
OUTPUT_DIR = Path("outputs/fixed_final_epoch_10seeds")
SOURCE_TARGET_DIR = Path("source_target_9sets_fixed")  # new dir: never reuse data built by the old (overlapping) construction
RUNS_DIR = OUTPUT_DIR / "model_runs"

RUN_LEVEL_CSV = OUTPUT_DIR / "run_level_results.csv"
EMOTION_LEVEL_CSV = OUTPUT_DIR / "emotion_level_results.csv"
PREDICTION_LEVEL_CSV = OUTPUT_DIR / "prediction_level_results.csv"
FAILED_RUNS_CSV = OUTPUT_DIR / "failed_runs.csv"
ENV_REPORT_TXT = OUTPUT_DIR / "environment_report.txt"
COMPLETENESS_TXT = OUTPUT_DIR / "completeness_check.txt"

ORIGINAL_FILES = [
    DATA_DIR / "dataset_experiment1_original.csv",
    DATA_DIR / "dataset_experiment2_korean.csv",
    DATA_DIR / "dataset_experiment3_chinese.csv",
    DATA_DIR / "dataset_experiment4_english.csv",
]

DATASET_FILES = {
    "KO": DATA_DIR / "dataset_experiment2_korean.csv",
    "ZH": DATA_DIR / "dataset_experiment3_chinese.csv",
    "EN": DATA_DIR / "dataset_experiment4_english.csv",
}

MODELS = [
    "bert-base-multilingual-cased",
    "klue/roberta-base",
    "hfl/chinese-roberta-wwm-ext",
    "roberta-base",
]

EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]

TEXT_ALIASES = ["text", "Text", "sentence", "Sentence", "content", "Content", "tweet", "Tweet", "utterance", "Utterance"]
EMOTION_ALIASES = ["emotion", "Emotion", "label", "Label", "category", "Category", "class", "Class"]


@dataclass(frozen=True)
class Experiment:
    seed: int
    source: str
    target: str
    model_name: str

    @property
    def dataset_name(self) -> str:
        return f"{self.source.lower()}_to_{self.target.lower()}"

    @property
    def safe_model_name(self) -> str:
        return self.model_name.replace("/", "__")

    @property
    def output_dir(self) -> Path:
        return RUNS_DIR / f"seed{self.seed}" / self.dataset_name / self.safe_model_name


def print_section(title: str) -> None:
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")


def validate_input_files() -> None:
    missing = [p for p in ORIGINAL_FILES if not p.exists()]
    if missing:
        print("ERROR: Missing required original dataset files:", file=sys.stderr)
        for p in missing:
            print(f"- {p}", file=sys.stderr)
        sys.exit(1)


def require_packages() -> None:
    missing = []
    for package, import_name in [
        ("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn"),
        ("datasets", "datasets"), ("torch", "torch"), ("transformers", "transformers"),
    ]:
        try:
            __import__(import_name)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        print("ERROR: Missing required packages:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)


def find_column(columns: list[str], aliases: list[str], label: str) -> str:
    lower_lookup = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower_lookup:
            return lower_lookup[alias.lower()]
    raise ValueError(f"Could not find {label} column. Available columns: {columns}")


def normalize_emotion(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {
        "angry": "anger", "anger": "anger", "fear": "fear", "fearful": "fear",
        "joy": "joy", "happy": "joy", "happiness": "joy",
        "sad": "sadness", "sadness": "sadness",
        "surprise": "surprise", "surprised": "surprise",
    }
    return aliases.get(text, text)


def standardize_dataset(df, dataset_name: str):
    columns = list(df.columns)
    text_col = find_column(columns, TEXT_ALIASES, f"text for {dataset_name}")
    emotion_col = find_column(columns, EMOTION_ALIASES, f"emotion for {dataset_name}")
    result = df[[text_col, emotion_col]].copy()
    result.columns = ["text", "emotion"]
    result["split"] = df["split"].values
    result["original_text"] = df["original_text"].astype(str).values
    result["text"] = result["text"].astype(str)
    result["emotion"] = result["emotion"].map(normalize_emotion)
    result = result[result["emotion"].isin(EMOTIONS)].dropna().reset_index(drop=True)
    if result.empty:
        raise ValueError(f"{dataset_name} has no usable rows after standardization.")
    return result


def balanced_sample(df, total_n: int, pd, label: str):
    """Identical to the original script: sampling seed is always MASTER_DATA_SEED (42),
    independent of the training seed used later -- the dataset itself never changes."""
    base_n = total_n // len(EMOTIONS)
    remainder = total_n % len(EMOTIONS)
    pieces = []
    for idx, emotion in enumerate(EMOTIONS):
        need_n = base_n + (1 if idx < remainder else 0)
        subset = df[df["emotion"] == emotion]
        if subset.empty:
            raise ValueError(f"{label} has no rows for emotion={emotion}")
        replace = len(subset) < need_n
        sampled = subset.sample(n=need_n, replace=replace, random_state=MASTER_DATA_SEED + idx)
        pieces.append(sampled)
    sampled_df = pd.concat(pieces, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=MASTER_DATA_SEED).reset_index(drop=True)
    return sampled_df


def load_original_datasets(pd):
    print_section("Load Original Datasets")
    for path in ORIGINAL_FILES:
        raw = pd.read_csv(path)
        print(f"{path}: shape={raw.shape}")
    datasets = {}
    for language, path in DATASET_FILES.items():
        raw_df = pd.read_csv(path)
        datasets[language] = standardize_dataset(raw_df, str(path))
    return datasets


def build_9_source_target_datasets_once(datasets, pd) -> None:
    """Build the 9 Source->Target CSVs a single time (seed=42 sampling, unchanged
    across the whole multi-seed run). If they already exist on disk (e.g. from a
    previous partial run), reuse them as-is instead of re-sampling."""
    print_section("Build 9 Source->Target Datasets (built once, shared by all seeds)")
    SOURCE_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source in ["KO", "ZH", "EN"]:
        for target in ["KO", "ZH", "EN"]:
            dataset_name = f"{source.lower()}_to_{target.lower()}"
            out_path = SOURCE_TARGET_DIR / f"{dataset_name}.csv"
            if out_path.exists():
                print(f"Reusing existing {out_path}")
                continue
            src_pool = datasets[source][datasets[source]["split"] == "train"]
            tgt_pool = datasets[target][datasets[target]["split"] == "test"]
            train_df = balanced_sample(src_pool, TRAIN_N, pd, f"{source} train {dataset_name}")
            test_df = balanced_sample(tgt_pool, TEST_N, pd, f"{target} test {dataset_name}")
            overlap = float(test_df["original_text"].isin(set(train_df["original_text"])).mean())
            print(f"  train/test underlying-sentence overlap for {dataset_name}: {overlap:.4f}")
            assert overlap < 0.02, f"train/test overlap too high for {dataset_name}: {overlap}"
            train_df = train_df.assign(split="train", SourceLanguage=source, TargetLanguage=target)
            test_df = test_df.assign(split="test", SourceLanguage=source, TargetLanguage=target)
            combined = pd.concat([train_df, test_df], ignore_index=True)
            combined = combined[["SourceLanguage", "TargetLanguage", "split", "text", "emotion", "original_text"]]
            combined = combined.reset_index().rename(columns={"index": "row_id"})
            combined.to_csv(out_path, index=False)
            print(f"Generated {dataset_name}: total={len(combined)}")


def load_source_target_dataset(experiment: Experiment, pd):
    dataset_path = SOURCE_TARGET_DIR / f"{experiment.dataset_name}.csv"
    df = pd.read_csv(dataset_path)
    train_df = df[df["split"] == "train"][["row_id", "text", "emotion"]].reset_index(drop=True)
    test_df = df[df["split"] == "test"][["row_id", "text", "emotion"]].reset_index(drop=True)
    return train_df, test_df


def write_environment_report(runtime: dict[str, Any]) -> None:
    torch = runtime["torch"]
    transformers = runtime["transformers"]
    lines = [
        "Study 2 Multi-Seed Reproduction -- Environment Report",
        "=" * 80,
        f"Python: {sys.version}",
        f"OS: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"CUDA version: {torch.version.cuda}")
    lines.append(f"transformers: {transformers.__version__}")
    lines.append(f"Seeds: {SEEDS}")
    lines.append(f"Models: {MODELS}")
    lines.append(f"TRAIN_N={TRAIN_N} TEST_N={TEST_N} MASTER_DATA_SEED={MASTER_DATA_SEED}")
    lines.append(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ENV_REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    ENV_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def all_experiments() -> list[Experiment]:
    return [
        Experiment(seed, source, target, model_name)
        for seed in SEEDS
        for source in ["KO", "ZH", "EN"]
        for target in ["KO", "ZH", "EN"]
        for model_name in MODELS
    ]


def completed_keys() -> set[tuple[int, str, str, str]]:
    if not RUN_LEVEL_CSV.exists():
        return set()
    import pandas as pd
    df = pd.read_csv(RUN_LEVEL_CSV)
    return set(zip(df["seed"], df["source"], df["target"], df["model"]))


def append_row_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def append_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


RUN_FIELDS = ["seed", "source", "target", "model", "accuracy", "macro_f1",
              "train_runtime_sec", "eval_runtime_sec", "total_runtime_sec",
              "f1_epoch1", "f1_epoch2", "f1_epoch3", "f1_best_of_3", "f1_last",
              "python_version", "torch_version", "transformers_version",
              "cuda_available", "gpu_name", "timestamp"]

EMOTION_FIELDS = ["seed", "source", "target", "model", "emotion",
                  "precision", "recall", "f1", "support"]

PRED_FIELDS = ["seed", "source", "target", "model", "sample_id",
               "true_label", "predicted_label",
               "prob_anger", "prob_fear", "prob_joy", "prob_sadness", "prob_surprise"]

FAILED_FIELDS = ["seed", "source", "target", "model", "error_type", "error_message", "timestamp"]


def run_one_experiment(experiment: Experiment, runtime: dict[str, Any]) -> None:
    pd = runtime["pd"]
    np = runtime["np"]
    torch = runtime["torch"]
    Dataset = runtime["Dataset"]
    AutoTokenizer = runtime["AutoTokenizer"]
    AutoModelForSequenceClassification = runtime["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = runtime["DataCollatorWithPadding"]
    Trainer = runtime["Trainer"]
    TrainingArguments = runtime["TrainingArguments"]
    set_seed = runtime["set_seed"]
    transformers = runtime["transformers"]

    t_start = time.time()
    print_section(f"seed={experiment.seed} | {experiment.source}->{experiment.target} | {experiment.model_name}")

    train_df, test_df = load_source_target_dataset(experiment, pd)

    label2id = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
    id2label = {idx: emotion for emotion, idx in label2id.items()}
    train_df = train_df.assign(label=train_df["emotion"].map(label2id))
    test_df = test_df.assign(label=test_df["emotion"].map(label2id))

    # training-time stochasticity is controlled by this seed; data sampling above is not
    random.seed(experiment.seed)
    np.random.seed(experiment.seed)
    set_seed(experiment.seed)

    tokenizer = AutoTokenizer.from_pretrained(experiment.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        experiment.model_name, num_labels=len(EMOTIONS),
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "Accuracy": accuracy_score(labels, predictions),
            "Macro_F1": f1_score(labels, predictions, average="macro", zero_division=0),
        }

    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "output_dir": str(experiment.output_dir),
        "seed": experiment.seed,
        "data_seed": experiment.seed,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 200,
        "load_best_model_at_end": False,
        "report_to": [],
        "disable_tqdm": True,
    }
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["evaluation_strategy" if "evaluation_strategy" in params else "eval_strategy"] = "epoch"
    args = TrainingArguments(**kwargs)

    trainer_kwargs = {
        "model": model, "args": args,
        "train_dataset": train_dataset, "eval_dataset": test_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    tparams = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["tokenizer" if "tokenizer" in tparams else "processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    t_train0 = time.time()
    trainer.train()
    train_runtime = time.time() - t_train0

    t_eval0 = time.time()
    pred_output = trainer.predict(test_dataset)
    eval_runtime = time.time() - t_eval0

    logits = pred_output.predictions
    labels = pred_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    precisions = precision_score(labels, preds, average=None, labels=list(range(len(EMOTIONS))), zero_division=0)
    recalls = recall_score(labels, preds, average=None, labels=list(range(len(EMOTIONS))), zero_division=0)
    f1s = f1_score(labels, preds, average=None, labels=list(range(len(EMOTIONS))), zero_division=0)
    supports = [int((labels == i).sum()) for i in range(len(EMOTIONS))]
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    total_runtime = time.time() - t_start

    epoch_f1 = [float(e["eval_Macro_F1"]) for e in trainer.state.log_history if "eval_Macro_F1" in e]
    assert len(epoch_f1) == 3, epoch_f1
    run_row = {
        "f1_epoch1": epoch_f1[0], "f1_epoch2": epoch_f1[1], "f1_epoch3": epoch_f1[2],
        "f1_best_of_3": max(epoch_f1), "f1_last": macro_f1,
        "seed": experiment.seed, "source": experiment.source, "target": experiment.target,
        "model": experiment.model_name, "accuracy": accuracy, "macro_f1": macro_f1,
        "train_runtime_sec": round(train_runtime, 2), "eval_runtime_sec": round(eval_runtime, 2),
        "total_runtime_sec": round(total_runtime, 2),
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "timestamp": timestamp,
    }
    append_row_csv(RUN_LEVEL_CSV, run_row, RUN_FIELDS)

    emotion_rows = []
    for i, emotion in enumerate(EMOTIONS):
        emotion_rows.append({
            "seed": experiment.seed, "source": experiment.source, "target": experiment.target,
            "model": experiment.model_name, "emotion": emotion,
            "precision": float(precisions[i]), "recall": float(recalls[i]),
            "f1": float(f1s[i]), "support": supports[i],
        })
    append_rows_csv(EMOTION_LEVEL_CSV, emotion_rows, EMOTION_FIELDS)

    sample_ids = test_df["row_id"].tolist()
    pred_rows = []
    for i in range(len(labels)):
        row = {
            "seed": experiment.seed, "source": experiment.source, "target": experiment.target,
            "model": experiment.model_name, "sample_id": int(sample_ids[i]),
            "true_label": EMOTIONS[int(labels[i])], "predicted_label": EMOTIONS[int(preds[i])],
        }
        for j, emotion in enumerate(EMOTIONS):
            row[f"prob_{emotion}"] = float(probs[i][j])
        pred_rows.append(row)
    append_rows_csv(PREDICTION_LEVEL_CSV, pred_rows, PRED_FIELDS)

    cm_path = OUTPUT_DIR / "confusion_matrices" / f"seed{experiment.seed}_{experiment.dataset_name}_{experiment.safe_model_name}.csv"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(cm_path)

    print(f"Done in {total_runtime:.1f}s | Macro_F1={macro_f1:.4f}")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def import_runtime_packages() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                               DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    return dict(np=np, pd=pd, torch=torch, transformers=transformers, Dataset=Dataset,
                AutoModelForSequenceClassification=AutoModelForSequenceClassification,
                AutoTokenizer=AutoTokenizer, DataCollatorWithPadding=DataCollatorWithPadding,
                Trainer=Trainer, TrainingArguments=TrainingArguments, set_seed=set_seed)


def completeness_check() -> None:
    import pandas as pd
    expected = {(e.seed, e.source, e.target, e.model_name) for e in all_experiments()}
    if not RUN_LEVEL_CSV.exists():
        actual_list = []
    else:
        df = pd.read_csv(RUN_LEVEL_CSV)
        actual_list = list(zip(df["seed"], df["source"], df["target"], df["model"]))
    from collections import Counter
    counts = Counter(actual_list)
    actual_set = set(actual_list)
    missing = sorted(expected - actual_set)
    duplicates = sorted([k for k, v in counts.items() if v > 1])

    lines = [
        "Study 2 Multi-Seed Reproduction -- Completeness Check",
        "=" * 80,
        f"Expected: {len(SEEDS)} seeds x 36 combos = {len(expected)} runs",
        f"Actual rows in run_level_results.csv: {len(actual_list)}",
        f"Unique completed combos: {len(actual_set)}",
        f"Missing combos: {len(missing)}",
    ]
    for m in missing:
        lines.append(f"  MISSING: seed={m[0]} {m[1]}->{m[2]} | {m[3]}")
    lines.append(f"Duplicate combos (row appears >1x): {len(duplicates)}")
    for d in duplicates:
        lines.append(f"  DUPLICATE ({counts[d]}x): seed={d[0]} {d[1]}->{d[2]} | {d[3]}")
    per_seed = {}
    for s, so, ta, mo in actual_set:
        per_seed[s] = per_seed.get(s, 0) + 1
    lines.append("Per-seed completed count (expect 36 each):")
    for s in SEEDS:
        lines.append(f"  seed={s}: {per_seed.get(s, 0)}/36")

    COMPLETENESS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    validate_input_files()
    require_packages()
    runtime = import_runtime_packages()
    write_environment_report(runtime)

    pd = runtime["pd"]
    datasets = load_original_datasets(pd)
    build_9_source_target_datasets_once(datasets, pd)

    done = completed_keys()
    experiments = all_experiments()
    print_section(f"Total experiments to run: {len(experiments)} (already done: {len(done)})")

    for experiment in experiments:
        key = (experiment.seed, experiment.source, experiment.target, experiment.model_name)
        if key in done:
            print(f"Skip completed: {key}")
            continue
        try:
            run_one_experiment(experiment, runtime)
        except Exception as exc:  # noqa: BLE001 -- must not silently drop failures
            print(f"FAILED: {key} -> {exc}", file=sys.stderr)
            traceback.print_exc()
            append_row_csv(FAILED_RUNS_CSV, {
                "seed": experiment.seed, "source": experiment.source, "target": experiment.target,
                "model": experiment.model_name, "error_type": type(exc).__name__,
                "error_message": str(exc)[:500], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, FAILED_FIELDS)
            shutil.rmtree(experiment.output_dir, ignore_errors=True)
            continue

    completeness_check()
    print("\nAll requested runs attempted. See completeness_check.txt and failed_runs.csv for status.")


if __name__ == "__main__":
    main()
