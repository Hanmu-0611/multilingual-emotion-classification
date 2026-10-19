import json
import os
import time

import pandas as pd
from openai import AuthenticationError
from openai import OpenAI
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_DIR = "emotion_data"
TRAIN_OUTPUT = os.path.join(DATA_DIR, "train_balanced_translated.csv")
TEST_OUTPUT = os.path.join(DATA_DIR, "test_balanced_translated.csv")
FAILED_LOG = os.path.join(DATA_DIR, "failed_translations.csv")

TRAIN_PER_GROUP = 500
TEST_PER_GROUP = 100
BATCH_SIZE = 3
TRANSLATION_MODEL = "gpt-4o-mini"
RANDOM_STATE = 42

TARGET_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise"]
LANGUAGES = ["ko", "en", "zh"]

LABEL2ID = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "surprise": 4,
}

TARGET_LANGUAGE_NAME = {
    "ko": "Korean",
    "zh": "Simplified Chinese",
    "en": "English",
}

TARGET_COLUMNS = {
    "ko": "text_ko",
    "zh": "text_zh",
    "en": "text_en",
}

SOURCE_FILES = {
    "ko": "korean_original.csv",
    "en": "english_original.csv",
    "zh": "chinese_original.csv",
}


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set.\n\n"
        "Run this first in terminal:\n"
        "export OPENAI_API_KEY='your_api_key'"
    )

client = OpenAI(timeout=30)


def load_original_data():
    frames = []

    for lang, filename in SOURCE_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing original file: {path}")

        df = pd.read_csv(path)
        required = {"text", "emotion", "language", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        df = df[["text", "emotion", "language", "label"]].copy()
        df["text"] = df["text"].astype(str)
        df["emotion"] = df["emotion"].astype(str)
        df["language"] = lang
        df["label"] = df["emotion"].map(LABEL2ID)
        df = df.dropna(subset=["text", "emotion", "language", "label"])
        df["label"] = df["label"].astype(int)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["emotion"].isin(TARGET_EMOTIONS)]
    all_df = all_df.drop_duplicates(subset=["text", "emotion", "language"])

    print("\nOriginal data statistics:")
    print(all_df.groupby(["language", "emotion"]).size())

    return all_df


def build_balanced_splits(all_df):
    train_parts = []
    test_parts = []

    for lang in LANGUAGES:
        for emotion in TARGET_EMOTIONS:
            group = all_df[
                (all_df["language"] == lang) &
                (all_df["emotion"] == emotion)
            ].copy()

            print(lang, emotion, "available:", len(group))

            if len(group) == 0:
                raise ValueError(f"No data for {lang}-{emotion}")

            test_n = min(TEST_PER_GROUP, len(group))
            test_part = group.sample(
                n=test_n,
                replace=False,
                random_state=RANDOM_STATE,
            )

            remaining = group.drop(test_part.index)
            train_part = remaining.sample(
                n=TRAIN_PER_GROUP,
                replace=len(remaining) < TRAIN_PER_GROUP,
                random_state=RANDOM_STATE,
            )

            train_parts.append(train_part)
            test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nTrain statistics:")
    print(train_df.groupby(["language", "emotion"]).size())

    print("\nTest statistics:")
    print(test_df.groupby(["language", "emotion"]).size())

    return train_df, test_df


def translate_batch(texts, source_language, target_language, retry=2):
    if source_language == target_language:
        return texts

    prompt = f"""
Translate each item into natural {TARGET_LANGUAGE_NAME[target_language]}.
Keep the same order and same number of items.
Do not add explanations.
Return the result in the required JSON schema.
"""

    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_batch",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "translations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": len(texts),
                                    "maxItems": len(texts),
                                }
                            },
                            "required": ["translations"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=0,
            )

            content = response.choices[0].message.content.strip()
            result = json.loads(content)["translations"]

            if isinstance(result, list) and len(result) == len(texts):
                return [str(item) for item in result]
        except Exception as exc:
            if isinstance(exc, AuthenticationError) or "Incorrect API key" in str(exc):
                raise
            print("Batch translation failed, retry:", attempt + 1, exc)
            time.sleep(3 + attempt * 2)

    raise ValueError("Batch translation failed.")


def translate_one_text(text, source_language, target_language, retry=2):
    if source_language == target_language:
        return text

    prompt = (
        f"Translate the user's text into natural "
        f"{TARGET_LANGUAGE_NAME[target_language]}. Return only the translation."
    )

    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": str(text)},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if isinstance(exc, AuthenticationError) or "Incorrect API key" in str(exc):
                raise
            print("Single translation failed, retry:", attempt + 1, exc)
            time.sleep(3 + attempt * 2)

    raise ValueError("Single translation failed.")


def log_failed_translation(text, source_language, target_language, reason):
    row = pd.DataFrame([{
        "source_language": source_language,
        "target_language": target_language,
        "text": str(text),
        "reason": str(reason),
    }])

    if os.path.exists(FAILED_LOG):
        row.to_csv(FAILED_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(FAILED_LOG, index=False)


def translate_batch_with_fallback(texts, source_language, target_language):
    try:
        return translate_batch(texts, source_language, target_language)
    except Exception as exc:
        if isinstance(exc, AuthenticationError) or "Incorrect API key" in str(exc):
            raise
        print("Batch failed. Falling back to single-item translation:", exc)
        translations = []

        for text in tqdm(texts, desc=f"single {source_language}->{target_language}", leave=False):
            try:
                translations.append(
                    translate_one_text(text, source_language, target_language)
                )
            except Exception as single_exc:
                print(
                    "Single item failed. Keeping original text and logging:",
                    single_exc,
                    flush=True,
                )
                log_failed_translation(
                    text,
                    source_language,
                    target_language,
                    single_exc,
                )
                translations.append(str(text))

        return translations


def ensure_translation_columns(df):
    for column in TARGET_COLUMNS.values():
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")
    return df


def add_translations_resume(df, output_path):
    if os.path.exists(output_path):
        print("\nResuming existing file:", output_path)
        df = pd.read_csv(output_path)
    else:
        df = df.copy()
        df = ensure_translation_columns(df)
        df.to_csv(output_path, index=False)

    df = ensure_translation_columns(df)
    df["text"] = df["text"].astype(str)

    for target_language, column in TARGET_COLUMNS.items():
        print("\nTarget column:", column)

        for source_language in LANGUAGES:
            mask = (
                (df["language"] == source_language) &
                (df[column].isna() | (df[column].astype(str).str.strip() == ""))
            )

            indices = df[mask].index.tolist()

            if not indices:
                print(f"{source_language}->{target_language} complete")
                continue

            if source_language == target_language:
                df.loc[indices, column] = df.loc[indices, "text"]
                df.to_csv(output_path, index=False)
                print(f"{source_language}->{target_language} copied original text")
                continue

            print(f"{source_language}->{target_language} remaining:", len(indices))

            for start in tqdm(range(0, len(indices), BATCH_SIZE)):
                batch_indices = indices[start:start + BATCH_SIZE]
                batch_texts = df.loc[batch_indices, "text"].tolist()

                print(
                    f"Translating {source_language}->{target_language}: "
                    f"{start + 1}-{min(start + BATCH_SIZE, len(indices))}/{len(indices)}",
                    flush=True,
                )

                translations = translate_batch_with_fallback(
                    batch_texts,
                    source_language,
                    target_language,
                )

                df.loc[batch_indices, column] = translations
                df.to_csv(output_path, index=False)
                print("Saved progress.", flush=True)

    print("\nFinished:", output_path)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_OUTPUT) and os.path.exists(TEST_OUTPUT):
        print("Translated train/test files already exist. Resuming translation only.")
        train_df = pd.read_csv(TRAIN_OUTPUT)
        test_df = pd.read_csv(TEST_OUTPUT)
    else:
        all_df = load_original_data()
        train_df, test_df = build_balanced_splits(all_df)

    train_df = add_translations_resume(train_df, TRAIN_OUTPUT)
    test_df = add_translations_resume(test_df, TEST_OUTPUT)

    print("\nFinal train columns:")
    print(train_df.columns.tolist())
    print("\nFinal test columns:")
    print(test_df.columns.tolist())


if __name__ == "__main__":
    main()
