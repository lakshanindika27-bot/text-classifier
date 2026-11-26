import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.baselines import load, scores
from src.baselines import make_models as tfidf_models
from src.data import SEED
from src.embed_eval import make_models as emb_models
from src.threshold import TARGET_PRECISION, oof_scores, pick_threshold
from src.tune_tfidf import char_vec

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "sms-spam"


def candidates(train, val):
    emb_tr = np.load("data/processed/emb_train.npy")
    emb_va = np.load("data/processed/emb_val.npy")
    for name, model in tfidf_models().items():
        if name.startswith("B0"):
            continue  # constant predictor, no threshold to tune
        yield name, "tfidf-word", model, train["text"], val["text"]
    yield ("B4 tfidf char + svm", "tfidf-char",
           make_pipeline(char_vec(), LinearSVC(C=1.0, random_state=SEED)),
           train["text"], val["text"])
    for name, model in emb_models().items():
        yield name, "nomic-embed-text", model, emb_tr, emb_va


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    train, val = load("train"), load("val")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, features, model, Xt, Xv in candidates(train, val):
        oof = oof_scores(model, Xt, train["y"], skf)
        thr = pick_threshold(train["y"].values, oof, TARGET_PRECISION)
        if thr is None:
            print(f"{name}: no threshold reaches precision {TARGET_PRECISION}")
            continue
        oof_pred = oof >= thr
        model.fit(Xt, train["y"])
        s = scores(model, Xv)
        pred = s >= thr
        tn, fp, fn, tp = confusion_matrix(val["y"], pred).ravel()
        clf = model[-1]
        params = {
            "features": features,
            "classifier": type(clf).__name__,
            "target_precision": TARGET_PRECISION,
            "cv_folds": 5,
            "seed": SEED,
            "threshold": round(float(thr), 4),
        }
        if hasattr(clf, "C"):
            params["C"] = clf.C
        metrics = {
            "oof_pr_auc": average_precision_score(train["y"], oof),
            "oof_precision": precision_score(train["y"], oof_pred),
            "oof_recall": recall_score(train["y"], oof_pred),
            "val_pr_auc": average_precision_score(val["y"], s),
            "val_precision": precision_score(val["y"], pred, zero_division=0),
            "val_recall": recall_score(val["y"], pred, zero_division=0),
            "val_f1": f1_score(val["y"], pred, zero_division=0),
            "val_fp": int(fp),
            "val_fn": int(fn),
        }
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"stage": "validation", "test_set_used": "false"})
            mlflow.log_params(params)
            mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics.items()})
        print(f"logged {name}: val_recall={metrics['val_recall']:.3f} "
              f"FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
