# Multilingual emotion classification: source-dataset condition and cross-lingual paths

Companion material for a manuscript accepted by the *Journal of The Korea Society of Computer and Information* (JKSCI).

This repository contains the code, the experimental design, and the text-free results of the study *"Differences in Multilingual Emotion Classification Performance by Source-Language Dataset Condition and Cross-Lingual Path"*.
It supersedes any earlier version of these results: the earlier evaluation selected checkpoints on the test set and, in Study 2, let training and test sentences overlap; every number here was recomputed with the final-epoch protocol, disjoint train/test pools, and 10 random seeds.

## What is (and is not) released
| Released | Not released (licences / size) |
|---|---|
| training, translation and analysis code | raw dataset text and its machine translations |
| per-run, per-source and per-emotion results (10 seeds) | |
| per-sentence predictions and class probabilities, keyed by `sample_id` (no text) | |
| sentence index with train/test split, emotion, origin language and the SHA-256 of each original sentence | |
| figures, analysis outputs, environment logs | |
| `results/analysis/extras/table2_dataset_characteristics_no_text.csv` (Table 2 numbers) and `A2_old_vs_new_*` (old-versus-new design comparison of Section V.3; the old runs themselves are not included) | |

The raw datasets must be obtained from their original sources (see `docs/data_sources.md`). The SHA-256 column in `data/sentence_index.csv` lets you check that a locally rebuilt sentence set matches ours. GPT-4o-mini translations cannot be regenerated bit-for-bit; see "Reproducing" below.

## Design in one paragraph
Nine thousand sentences (3 source datasets x 5 emotions x 600) are split into 7,500 training and 1,500 test sentences and rendered in Korean, Chinese and English with GPT-4o-mini (so every language version contains the same sentences). **Study 1** crosses TargetLanguage (KO/ZH/EN) x Model Strategy (multilingual mBERT vs. language-specific model) x SourceLanguage (origin dataset of the test sentence); an Original (mixed-language) mBERT run is a reference condition. **Study 2** trains on one language version and tests on another (9 paths x 4 models); all paths use the same 2,500 training and 500 test sentences. **Study 3** compares emotion-wise stability across conditions and analyses sadness errors. Supplementary analyses: independent translation check (NLLB-200, LaBSE, chrF), an NLLB-translated re-run of Study 2 (mBERT), text-length control, and the effect of picking the best epoch on the test set. Every configuration is run with seeds 42, 123, 2026, 7, 17, 29, 53, 71, 101, 31415; the data split is fixed (sampling seed 42), so seeds vary only training randomness.

## Layout
```
code/data_preparation/   translation (GPT-4o-mini) and construction of the experiment files
code/training/           Study 1, Study 2 and NLLB re-run training scripts (Hugging Face Trainer)
code/translation_check/  NLLB-200 re-translation, back-translation, LaBSE and chrF
code/analysis/           ANOVA / stability / extra analyses and figure scripts
data/                    sentence_index.csv, study2_train_test_row_ids.csv (no text)
results/study1|study2|study2_nllb_mbert/   run-, source-, emotion- and prediction-level results
results/translation_check/                 summaries (no text)
results/analysis/        tables produced by the analysis scripts
figures/                 manuscript figures
docs/                    data sources, label mapping, environment
SHA256SUMS.txt           checksums of every file in this repository
```

## Result files
* `run_level_results.csv` — one row per training run: `seed, target/source, model, accuracy, macro_f1, f1_epoch1..3, f1_best_of_3, f1_last, runtimes, software versions, GPU`.
* `source_level_results.csv` (Study 1) — Macro-F1 per run and per SourceLanguage subset (500 test sentences each).
* `emotion_level_results.csv` — precision / recall / F1 / support per emotion.
* `prediction_level_results.csv.gz` — per test sentence: `sample_id`, true and predicted label, class probabilities (read with `pandas.read_csv`).
* `confusion_matrices/` — one 5x5 matrix per run.

## Reproducing
1. Obtain the three raw datasets (`docs/data_sources.md`) and map them to the five emotions with the rules in `docs/label_mapping.md` (the original mapping script was not preserved; the rules were reconstructed and verified row-by-row against the processed files).
2. Create the translated experiment files with `code/data_preparation/01_prepare_translated_splits.py` and `03_make_experiment_datasets.py` (needs an OpenAI API key in `OPENAI_API_KEY`; model name `gpt-4o-mini`, no dated snapshot was recorded, so regenerated translations will differ slightly from ours). The resulting `emotion_data/dataset_experiment{1..4}_*.csv` must have the row order of `data/sentence_index.csv`.
3. Train: `python code/training/study1_final_epoch_10seeds.py`, `python code/training/study2_final_epoch_10seeds.py` (resumable; 70 and 360 runs).
4. Analyse: `python code/analysis/analyze_study1.py results/study1 out/s1`, `analyze_study2.py results/study2 out/s2`, `analyze_study3.py results/study2 results/study1 out/s3`, `analyze_extras.py . out/extras`, `analyze_length_control.py . out/len <emotion_data_dir>`, `make_figures.py . out/figures`.

The analyses in `results/` can be re-run from the released CSVs without any text (except the length control, which needs the sentence text).

## Environment (Colab, all training and translation checks)
Python 3.13.15, PyTorch 2.11.0+cu130 (CUDA 13.0), Transformers 5.17.0, NVIDIA RTX PRO 6000 Blackwell (96 GB); sentence-transformers 5.7.0, sacrebleu 2.6.0. Analyses were run locally with SciPy 1.17 and statsmodels 0.15. Hyper-parameters: max length 128, learning rate 2e-5, batch size 16, 3 epochs, weight decay 0.01, `Trainer` defaults otherwise (AdamW, linear decay, no warm-up, no class weights).

## Licences
Code: MIT (`LICENSE`). Results, figures and documentation: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The underlying datasets keep their own licences (KorEmpatheticDialogues: CC-BY-NC-4.0; GoEmotions: Apache-2.0 per the Hugging Face card; CLUEEmotionAnalysis2020: none found). Attribution, modifications and permitted use: `docs/DATA_SOURCES_AND_LICENSES.md`.

## Citation
Hai-Tao Han, "Differences in Multilingual Emotion Classification Performance by Source-Language Dataset Condition and Cross-Lingual Path," Journal of The Korea Society of Computer and Information (accepted; volume, pages and DOI to be added).
