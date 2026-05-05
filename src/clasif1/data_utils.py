from __future__ import annotations
import pandas as pd
import config


def load_dataset():
    df = pd.read_csv(config.DATASET_PATH)
    return df

def map_labels(df):
    df = df.copy()
    df["label"] = df[config.LABEL_COLUMN].map(config.LABEL2ID)
    return df

def select_columns(df):
    out = pd.DataFrame({
        "text": df[config.TEXT_COLUMN].astype(str),
        "label": df["label"].astype(int),
        "annotation": df[config.LABEL_COLUMN],
        "topic": df["topic"]
    })
    return out

def load_splits():
    df = load_dataset()
    df = map_labels(df)
    
    splits = {
        "train": df[df[config.SPLIT_COLUMN] == "train"],
        "val": df[df[config.SPLIT_COLUMN] == "val"],
        "test": df[df[config.SPLIT_COLUMN] == "test"]
    }
    
    train_df = select_columns(splits["train"]).reset_index(drop=True)
    val_df = select_columns(splits["val"]).reset_index(drop=True)
    test_df = select_columns(splits["test"]).reset_index(drop=True)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()
    print(train_df.head())
    print(val_df.head())
    print(test_df.head())
    