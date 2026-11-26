import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.baselines import load, scores
from src.data import SEED
from src.embed_eval import make_models as emb_models
from src.threshold import TARGET_PRECISION, oof_scores, pick_threshold
from src.tune_tfidf import char_vec


def main():
    train, val = load("train"), load("val")
    Xtr = np.load("data/processed/emb_train.npy")
    Xva = np.load("data/processed/emb_val.npy")
    candidates = {
        "B4 tfidf char + svm (C=1.0)": (
            make_pipeline(char_vec(), LinearSVC(C=1.0, random_state=SEED)),
            train["text"], val["text"]),
        "B3b embeddings + svm": (
            emb_models()["B3b embeddings + linear svm"], Xtr, Xva),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, (model, Xt, Xv) in candidates.items():
        oof = oof_scores(model, Xt, train["y"], skf)
        thr = pick_threshold(train["y"].values, oof, TARGET_PRECISION)
        oof_pred = oof >= thr
        model.fit(Xt, train["y"])
        s = scores(model, Xv)
        pred = s >= thr
        tn, fp, fn, tp = confusion_matrix(val["y"], pred).ravel()
        rows.append({
            "model": name,
            "oof_pr_auc": round(average_precision_score(train["y"], oof), 4),
            "oof_recall@p98": round(recall_score(train["y"], oof_pred), 3),
            "val_pr_auc": round(average_precision_score(val["y"], s), 4),
            "val_precision": round(precision_score(val["y"], pred, zero_division=0), 3),
            "val_recall": round(recall_score(val["y"], pred, zero_division=0), 3),
            "val_f1": round(f1_score(val["y"], pred, zero_division=0), 3),
            "FP": int(fp),
            "FN": int(fn),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("reports/final_candidates_val.csv", index=False)


if __name__ == "__main__":
    main()
