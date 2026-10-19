# Environment of the reported runs
Taken from the logs written by the runs themselves (`results/*/environment_report.txt` and the per-run columns of `run_level_results.csv`).
* Python 3.13.15 (Linux 6.6.122, glibc 2.39), PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 5.17.0
* GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
* Translation check: NLLB-200 distilled 1.3B (`facebook/nllb-200-distilled-1.3B`), LaBSE (`sentence-transformers/LaBSE`), sentence-transformers 5.7.0, sacrebleu 2.6.0; beam size 4, max 256 new tokens
* Runs: Study 1 on 2026-10-03; Study 2 on 2026-10-03; NLLB re-run on 2026-10-03; translation check on 2026-10-03
* GPT-4o-mini translations: OpenAI Chat Completions API, model name `gpt-4o-mini` (no dated snapshot or SDK version recorded), temperature 0, batches of 3 sentences with a strict JSON schema, single-sentence fallback; produced between 2026-06-03 and 2026-06-04
