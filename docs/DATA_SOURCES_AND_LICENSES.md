# Data sources, attribution, modifications and licence notice

**This repository does not contain any dataset text or translation.** The original datasets must be obtained from the sources below under their own licences. Licence information was checked on 2026-10-19 and is not legal advice.

## 1. Sources, attribution and licence status

| Dataset | Language | Source | Licence found (2026-10-19) | Status for redistribution |
|---|---|---|---|---|
| KorEmpatheticDialogues | KO | https://huggingface.co/datasets/passing2961/KorEmpatheticDialogues (commit `5373a3e`) | **CC-BY-NC-4.0** (dataset card metadata) | Allowed with attribution, **non-commercial use only**, and an indication of changes. |
| GoEmotions | EN | https://github.com/google-research/google-research/tree/master/goemotions ; Hugging Face mirror `google-research-datasets/go_emotions` | The Hugging Face card states **Apache-2.0**; the repository code is Apache-2.0. The repository page itself names **no licence for the data** and gives no terms for the underlying Reddit comments. | Probably allowed, **not fully verified** (Reddit terms unclear). |
| CLUEEmotionAnalysis2020 | ZH | https://github.com/CLUEbenchmark/CLUEmotionAnalysis2020 (master, files dated 2020-01-29) | **No licence file and no licence statement** (GitHub reports `license: None`; the README gives only a citation). Texts are social-media posts collected by hashtag. | **Not cleared.** Treat as all rights reserved; do not publish. |

Required citations:
* Lee et al., "Language Model Evaluation Based on Korean-English Empathetic Dialogue Datasets and Personality", HLT 2023 (KorEmpatheticDialogues, a DeepL translation of EmpatheticDialogues; Rashkin et al., ACL 2019).
* Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", ACL 2020.
* Li, Long, Lu and Li, "Emotion Corpus Construction Based on Selection from Hashtags", LREC 2016 (the corpus behind CLUEEmotionAnalysis2020); distributed by the CLUE benchmark.

Original files and their SHA-256 (first 12 hex characters) are listed in `docs/data_sources.md`.

## 2. Modifications made by this study

1. **Label mapping (selection of classes):** only joy, sadness, anger, fear, surprise are used; other labels are excluded, GoEmotions annotation rows with two or more target emotions are dropped, identical (text, emotion) pairs are merged (details: `docs/label_mapping.md`).
2. **Sampling:** per language and emotion, 100 test and 500 training sentences (random_state=42), 9,000 sentences in total.
3. **Machine translation:** every sentence was translated into the other languages (Korean, Chinese, English) with GPT-4o-mini (not redistributed) and, for the robustness re-run, with NLLB-200 (not redistributed). The translations are **derivative works** of the originals; the translation quality was not validated by human annotators.
4. **No other text editing.** The experiment files hold the source sentences unchanged apart from these steps.

## 3. Intended use

Anyone rebuilding the experiment files must follow the licences of the sources: KorEmpatheticDialogues is non-commercial (CC-BY-NC-4.0), and translated files inherit the terms of the sources they were derived from. Keep the attributions above and mention that the data were selected and machine translated.

## 4. What is released

Code, row identifiers and SHA-256 hashes of the original sentences (`data/`), and text-free results. Sentence texts and the GPT-4o-mini / NLLB-200 translations are not redistributed because of the source licences (CLUEEmotionAnalysis2020 states none; the Reddit terms behind GoEmotions were not verified). Readers download the three original datasets from the links above and rebuild the subset with `code/data_preparation/`; the SHA-256 column of `data/sentence_index.csv` verifies the result.
