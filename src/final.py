import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.baselines import load, scores
from src.data import SEED
from src.threshold import TARGET_PRECISION, oof_scores, pick_threshold
from src.tune_tfidf import char_vec

VERSION = "tfidf-char-svm-v1"
N_BOOT = 2000
ARTIFACT = Path("models/spam_model.joblib")


def build():
    return make_pipeline(char_vec(), LinearSVC(C=1.0, random_state=SEED))


def bootstrap_ci(y, pred, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    funcs = {"precision": precision_score, "recall": recall_score, "f1": f1_score}
    stats = {k: [] for k in funcs}
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        for k, f in funcs.items():
            stats[k].append(f(y[b], pred[b], zero_division=0))
    return {k: [round(float(np.percentile(v, 2.5)), 3),
                round(float(np.percentile(v, 97.5)), 3)] for k, v in stats.items()}


def main():
    train, test = load("train"), load("test")
    model = build()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = oof_scores(model, train["text"], train["y"], skf)
    thr = pick_threshold(train["y"].values, oof, TARGET_PRECISION)
    model.fit(train["text"], train["y"])

    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model, "threshold": thr, "version": VERSION,
        "target_precision": TARGET_PRECISION,
        "sklearn_version": sklearn.__version__,
    }, ARTIFACT)

    s = scores(model, test["text"])
    y = test["y"].values
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    result = {
        "model": VERSION, "threshold": round(float(thr), 4),
        "n_test": int(len(y)), "n_spam": int(y.sum()),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "ci95": bootstrap_ci(y, pred),
    }
    print(json.dumps(result, indent=2))
    Path("reports/test_results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
