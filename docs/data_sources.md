# Data sources

| Dataset | Language | Source | Version used | Licence |
|---|---|---|---|---|
| KorEmpatheticDialogues | KO | https://huggingface.co/datasets/passing2961/KorEmpatheticDialogues | commit `5373a3e` (2024-11-06, head of the repository) ; 24,847 dialogues (19,531 / 2,769 / 2,547) | CC-BY-NC-4.0 (as stated on the dataset page) |
| CLUEEmotionAnalysis2020 | ZH | https://github.com/CLUEbenchmark/CLUEmotionAnalysis2020 | master, commit `ff1bc94` (2020-01-29, head); local archive `cluemotion.zip` SHA-256 `436bb7b283ac...` | none found (no LICENSE file, no statement in README; checked 2026-10-19) — not cleared |
| GoEmotions (full dataset) | EN | https://github.com/google-research/google-research/tree/master/goemotions (raw CSVs) | 3 files, 211,225 annotation rows; SHA-256 (first 12 hex) `cac049036bad` (goemotions_1.csv), `f699ecc5aa42` (goemotions_2.csv), `467f1e7191af` (goemotions_3.csv) | Hugging Face card: Apache-2.0; the repository page names no data licence (checked 2026-10-19) — see DATA_SOURCES_AND_LICENSES.md |

Files were obtained by 2026-06-01 (the archive of the raw files was exported on that date); the repository pages were re-checked on 2026-10-04. Because the latest commits of both repositories pre-date 2026-06-01, the commits above are the versions available at download time.
KorEmpatheticDialogues is a DeepL translation of EmpatheticDialogues (Rashkin et al., ACL 2019); cite Lee et al., "Language Model Evaluation Based on Korean-English Empathetic Dialogue Datasets and Personality", HLT 2023, as requested on the dataset page.
Korean raw files: `korean_train.csv` SHA-256 `a1718db44de8...`, `korean_validation.csv` `e817d946799f...`, `korean_test.csv` `b47b0233d100...`.

Full attribution, modification and licence notice: `DATA_SOURCES_AND_LICENSES.md`.
