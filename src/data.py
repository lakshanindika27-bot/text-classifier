import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
RAW = Path("data/raw/SMSSpamCollection")
OUT = Path("data/processed")


def load_raw(path=RAW):
    df = pd.read_csv(
        path, sep="\t", header=None, names=["label", "text"],
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def dedupe(df):
    """Drop duplicate messages (case/space-insensitive).
    Returns (deduped_df, number_of_texts_with_conflicting_labels)."""
    key = df["text"].str.strip().str.lower()
    conflicts = int(df.groupby(key)["y"].nunique().gt(1).sum())
    out = df.loc[~key.duplicated()].reset_index(drop=True)
    return out, conflicts


def split(df, seed=SEED):
    """Stratified 70/15/15 train/val/test split."""
    train, rest = train_test_split(
        df, test_size=0.30, stratify=df["y"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest["y"], random_state=seed
    )
    return train, val, test


def main():
    raw = load_raw()
    df, conflicts = dedupe(raw)
    print(f"raw: {len(raw)} rows, spam {raw['y'].mean():.1%}")
    print(f"after dedupe: {len(df)} rows ({len(raw) - len(df)} removed), "
          f"spam {df['y'].mean():.1%}, label conflicts: {conflicts}")
    train, val, test = split(df)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, spam {part['y'].mean():.1%} "
              f"({int(part['y'].sum())} spam)")


if __name__ == "__main__":
    main()
