import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from src.baselines import load, scores
from src.data import SEED
from src.threshold import TARGET_PRECISION, oof_scores, pick_threshold


def make_models():
    return {
        "B3a embeddings + logreg": make_pipeline(
            Normalizer(), LogisticRegression(C=10, max_iter=2000, random_state=SEED)),
        "B3b embeddings + linear svm": make_pipeline(
            Normalizer(), LinearSVC(random_state=SEED)),
    }


def main():
    train, val = load("train"), load("val")
    Xtr = np.load("data/processed/emb_train.npy")
    Xva = np.load("data/processed/emb_val.npy")
    assert len(Xtr) == len(train) and len(Xva) == len(val)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in make_models().items():
        oof = oof_scores(model, Xtr, train["y"], skf)
        thr = pick_threshold(train["y"].values, oof, TARGET_PRECISION)
        if thr is None:
            print(f"{name}: no threshold reaches precision {TARGET_PRECISION}")
            continue
        oof_pred = oof >= thr
        model.fit(Xtr, train["y"])
        s = scores(model, Xva)
        pred = s >= thr
        tn, fp, fn, tp = confusion_matrix(val["y"], pred).ravel()
        rows.append({
            "model": name,
            "threshold": round(thr, 3),
            "oof_precision": round(precision_score(train["y"], oof_pred), 3),
            "oof_recall": round(recall_score(train["y"], oof_pred), 3),
            "val_precision": round(precision_score(val["y"], pred, zero_division=0), 3),
            "val_recall": round(recall_score(val["y"], pred, zero_division=0), 3),
            "val_f1": round(f1_score(val["y"], pred, zero_division=0), 3),
            "val_pr_auc": round(average_precision_score(val["y"], s), 3),
            "FP": int(fp),
            "FN": int(fn),
        })
    out = pd.DataFrame(rows)
    print(f"Target precision (on train out-of-fold): {TARGET_PRECISION}")
    print(out.to_string(index=False))
    out.to_csv("reports/embed_val.csv", index=False)


if __name__ == "__main__":
    main()
