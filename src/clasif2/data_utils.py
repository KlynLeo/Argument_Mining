from __future__ import annotations

import pandas as pd

import config


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(config.DATASET_PATH)


def filter_argumentative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df[df[config.LABEL_COLUMN].isin(config.LABEL2ID)]


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df[config.LABEL_COLUMN].map(config.LABEL2ID)
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": df[config.TEXT_COLUMN].astype(str),
            "label": df["label"].astype(int),
            "annotation": df[config.LABEL_COLUMN],
            "topic": df["topic"],
        }
    )


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset()
    df = filter_argumentative(df)
    df = map_labels(df)

    splits = {
        "train": df[df[config.SPLIT_COLUMN] == "train"],
        "val": df[df[config.SPLIT_COLUMN] == "val"],
        "test": df[df[config.SPLIT_COLUMN] == "test"],
    }

    train_df = select_columns(splits["train"]).reset_index(drop=True)
    val_df = select_columns(splits["val"]).reset_index(drop=True)
    test_df = select_columns(splits["test"]).reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()
    print("train:")
    print(train_df.head())
    print("val:")
    print(val_df.head())
    print("test:")
    print(test_df.head())
