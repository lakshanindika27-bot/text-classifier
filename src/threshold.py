import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.baselines import load, make_models, scores
from src.data import SEED

TARGET_PRECISION = 0.98


def oof_scores(model, X, y, cv):
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    s = cross_val_predict(model, X, y, cv=cv, method=method)
    return s[:, 1] if s.ndim == 2 else s


def pick_threshold(y, s, target):
    """Lowest-cost threshold: max recall subject to precision >= target."""
    p, r, t = precision_recall_curve(y, s)
    ok = np.where(p[:-1] >= target)[0]
    if len(ok) == 0:
        return None
    return float(t[ok[np.argmax(r[:-1][ok])]])


def main():
    train, val = load("train"), load("val")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in make_models().items():
        if name.startswith("B0"):
            continue
        oof = oof_scores(model, train["text"], train["y"], skf)
        thr = pick_threshold(train["y"].values, oof, TARGET_PRECISION)
        if thr is None:
            print(f"{name}: no threshold reaches precision {TARGET_PRECISION}")
            continue
        oof_pred = oof >= thr
        model.fit(train["text"], train["y"])
        pred = scores(model, val["text"]) >= thr
        tn, fp, fn, tp = confusion_matrix(val["y"], pred).ravel()
        rows.append({
            "model": name,
            "threshold": round(thr, 3),
            "oof_precision": round(precision_score(train["y"], oof_pred), 3),
            "oof_recall": round(recall_score(train["y"], oof_pred), 3),
            "val_precision": round(precision_score(val["y"], pred, zero_division=0), 3),
            "val_recall": round(recall_score(val["y"], pred, zero_division=0), 3),
            "val_f1": round(f1_score(val["y"], pred, zero_division=0), 3),
            "FP": int(fp),
            "FN": int(fn),
        })
    out = pd.DataFrame(rows)
    print(f"Target precision (on train out-of-fold): {TARGET_PRECISION}")
    print(out.to_string(index=False))
    out.to_csv("reports/threshold_val.csv", index=False)


if __name__ == "__main__":
    main()
