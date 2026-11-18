import pandas as pd

from src.data import dedupe, split


def make_df(n=400):
    rows = [
        {"label": "spam" if i % 8 == 0 else "ham",
         "text": f"message number {i}",
         "y": int(i % 8 == 0)}
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_dedupe_ignores_case_and_spaces():
    df = pd.DataFrame({
        "label": ["ham", "ham", "spam"],
        "text": ["Hello there", " hello there ", "win cash"],
        "y": [0, 0, 1],
    })
    out, conflicts = dedupe(df)
    assert len(out) == 2
    assert conflicts == 0


def test_splits_do_not_overlap():
    train, val, test = split(make_df())
    a, b, c = set(train["text"]), set(val["text"]), set(test["text"])
    assert not (a & b) and not (a & c) and not (b & c)


def test_split_sizes_and_stratification():
    df = make_df()
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.70) < 0.02
    for part in (train, val, test):
        assert abs(part["y"].mean() - df["y"].mean()) < 0.02


def test_split_is_reproducible():
    df = make_df()
    t1, _, _ = split(df)
    t2, _, _ = split(df)
    assert list(t1["text"]) == list(t2["text"])
