# Mapping of original labels to the five emotions
The original mapping script was not preserved. The rules below were reconstructed by comparing the raw files with the processed files and reproduce them row by row (e.g. 21,049 / 21,049 rows for GoEmotions).

Unified classes: joy=0, sadness=1, anger=2, fear=3, surprise=4.

| Dataset | Original labels (units) | Rule | Pool after mapping (joy / sadness / anger / fear / surprise) |
|---|---|---|---|
| KorEmpatheticDialogues | 32 emotions (24,847 dialogues; the `situation` sentence is used) | joyful→joy, sad→sadness, angry→anger, afraid→fear, surprised→surprise; the other 27 labels are excluded, none merged | 4,516 (763 / 851 / 847 / 786 / 1,269) |
| CLUEEmotionAnalysis2020 | 7 labels (39,661 posts) | happiness→joy; sadness, anger, fear, surprise kept; like and disgust excluded | 30,245 (9,959 / 14,052 / 4,562 / 661 / 1,011) |
| GoEmotions | 27 emotions + neutral (211,225 annotation rows) | an annotation row is kept only if exactly one of {anger, fear, joy, sadness, surprise} is marked; 756 rows with two or more are dropped; the other 22 emotions and neutral form no class; identical (text, emotion) pairs are merged | 21,049 rows from 19,165 distinct comments (5,542 / 4,407 / 5,503 / 1,958 / 3,639) |

Label noise: 1,804 GoEmotions comments received different target labels from different annotators and remain as conflicting rows; 7 (KO) and 15 (ZH) texts carry conflicting labels.
Sampling (`01_prepare_translated_splits.py`, random_state=42): per language and emotion, 100 test sentences are drawn first and 500 training sentences from the remainder; the smallest group is Chinese fear (661 sentences).
