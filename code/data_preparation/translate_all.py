import json
import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


TRANSLATION_MODEL = "gpt-4o-mini"
BATCH_SIZE = 5

DATA_DIR = "emotion_data"

FILES = [
    {
        "input": "korean_original.csv",
        "output": "korean_original_translated.csv",
        "source_language": "ko",
    },
    {
        "input": "english_original.csv",
        "output": "english_original_translated.csv",
        "source_language": "en",
    },
    {
        "input": "chinese_original.csv",
        "output": "chinese_original_translated.csv",
        "source_language": "zh",
    },
]

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


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set.\n\n"
        "Mac/Linux terminal:\n"
        "export OPENAI_API_KEY='your_api_key'\n\n"
        "Windows PowerShell:\n"
        "$env:OPENAI_API_KEY='your_api_key'"
    )

client = OpenAI()


def translate_batch(texts, source_language, target_language, retry=5):
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

            print("Returned count mismatch, retrying:", len(result), "expected:", len(texts))
        except Exception as exc:
            print("Translation failed, retry:", attempt + 1, exc)
            time.sleep(3 + attempt * 2)

    raise ValueError("Translation failed. Check API key, model, or response format.")


def translate_one_text(text, source_language, target_language, retry=5):
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
            print("Single translation failed, retry:", attempt + 1, exc)
            time.sleep(3 + attempt * 2)

    raise ValueError("Single translation failed. Check API key or model.")


def translate_batch_with_fallback(texts, source_language, target_language):
    try:
        return translate_batch(texts, source_language, target_language)
    except Exception as exc:
        print("Batch translation failed. Falling back to single-item translation:", exc)
        return [
            translate_one_text(text, source_language, target_language)
            for text in tqdm(texts, desc=f"single {source_language}->{target_language}", leave=False)
        ]


def ensure_columns(df):
    for col in TARGET_COLUMNS.values():
        if col not in df.columns:
            df[col] = ""
    return df


def save_progress(df, output_path):
    df.to_csv(output_path, index=False)


def translate_one_file(input_path, output_path, source_language):
    print("\n==============================")
    print("Processing:", input_path)
    print("==============================")

    if os.path.exists(output_path):
        print("Found existing translated file. Resuming:", output_path)
        df = pd.read_csv(output_path)
    else:
        df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError(f"Missing required column 'text' in {input_path}")

    df = ensure_columns(df)
    df["text"] = df["text"].astype(str)
    save_progress(df, output_path)

    for target_language, col_name in TARGET_COLUMNS.items():
        print("\nTarget column:", col_name)

        if source_language == target_language:
            empty_mask = df[col_name].isna() | (df[col_name].astype(str).str.strip() == "")
            df.loc[empty_mask, col_name] = df.loc[empty_mask, "text"]
            save_progress(df, output_path)
            print("Source equals target. Original text copied:", col_name)
            continue

        mask = df[col_name].isna() | (df[col_name].astype(str).str.strip() == "")
        idx_list = df[mask].index.tolist()
        print(f"{source_language} -> {target_language} remaining:", len(idx_list))

        if not idx_list:
            continue

        for start in tqdm(range(0, len(idx_list), BATCH_SIZE)):
            batch_idx = idx_list[start : start + BATCH_SIZE]
            batch_texts = df.loc[batch_idx, "text"].tolist()
            print(
                f"Translating {source_language}->{target_language}: "
                f"{start + 1}-{min(start + BATCH_SIZE, len(idx_list))}/{len(idx_list)}",
                flush=True,
            )

            batch_result = translate_batch_with_fallback(
                batch_texts,
                source_language,
                target_language,
            )

            df.loc[batch_idx, col_name] = batch_result
            save_progress(df, output_path)
            print("Saved progress.", flush=True)

    print("Finished:", output_path)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for item in FILES:
        input_path = os.path.join(DATA_DIR, item["input"])
        output_path = os.path.join(DATA_DIR, item["output"])

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        translate_one_file(
            input_path=input_path,
            output_path=output_path,
            source_language=item["source_language"],
        )

    print("\nAll translations finished.")


if __name__ == "__main__":
    main()
