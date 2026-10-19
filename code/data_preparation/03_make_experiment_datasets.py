import os

import pandas as pd


DATA_DIR = "emotion_data"

TRAIN_PATH = os.path.join(DATA_DIR, "train_balanced_translated.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_balanced_translated.csv")

OUTPUTS = {
    "original": {
        "text_column": "text",
        "output": "dataset_experiment1_original.csv",
    },
    "korean": {
        "text_column": "text_ko",
        "output": "dataset_experiment2_korean.csv",
    },
    "chinese": {
        "text_column": "text_zh",
        "output": "dataset_experiment3_chinese.csv",
    },
    "english": {
        "text_column": "text_en",
        "output": "dataset_experiment4_english.csv",
    },
}


def load_split(path, split_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df["split"] = split_name
    return df


def validate_required_columns(df):
    required = {
        "text",
        "text_ko",
        "text_zh",
        "text_en",
        "emotion",
        "language",
        "label",
        "split",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def make_experiment_file(df, name, text_column, output_name):
    output_df = pd.DataFrame({
        "text": df[text_column].astype(str),
        "original_text": df["text"].astype(str),
        "original_language": df["language"].astype(str),
        "emotion": df["emotion"].astype(str),
        "label": df["label"].astype(int),
        "split": df["split"].astype(str),
    })

    empty_count = int(
        output_df["text"].isna().sum()
        + (output_df["text"].astype(str).str.strip() == "").sum()
    )

    if empty_count > 0:
        raise ValueError(f"{name} has empty text rows: {empty_count}")

    output_path = os.path.join(DATA_DIR, output_name)
    output_df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Rows:", len(output_df))
    print("Split:")
    print(output_df["split"].value_counts())
    print("Original language x emotion:")
    print(output_df.groupby(["original_language", "emotion"]).size())


def main():
    train_df = load_split(TRAIN_PATH, "train")
    test_df = load_split(TEST_PATH, "test")

    df = pd.concat([train_df, test_df], ignore_index=True)
    validate_required_columns(df)

    for name, config in OUTPUTS.items():
        make_experiment_file(
            df=df,
            name=name,
            text_column=config["text_column"],
            output_name=config["output"],
        )


if __name__ == "__main__":
    main()
