import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data import SEED


def load(name):
    return pd.read_csv(f"data/processed/{name}.csv")


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def make_models():
    return {
        "B0 majority class": DummyClassifier(strategy="most_frequent"),
        "B1 tfidf + logreg": make_pipeline(
            tfidf(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "B2a tfidf + naive bayes": make_pipeline(tfidf(), MultinomialNB()),
        "B2b tfidf + linear svm": make_pipeline(
            tfidf(), LinearSVC(random_state=SEED)),
    }


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    train, val = load("train"), load("val")
    f1_scorer = make_scorer(f1_score, zero_division=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in make_models().items():
        cv = cross_val_score(
            model, train["text"], train["y"], cv=skf, scoring=f1_scorer)
        model.fit(train["text"], train["y"])
        pred = model.predict(val["text"])
        s = scores(model, val["text"])
        tn, fp, fn, tp = confusion_matrix(val["y"], pred).ravel()
        rows.append({
            "model": name,
            "cv_f1": round(cv.mean(), 3),
            "val_precision": round(precision_score(val["y"], pred, zero_division=0), 3),
            "val_recall": round(recall_score(val["y"], pred, zero_division=0), 3),
            "val_f1": round(f1_score(val["y"], pred, zero_division=0), 3),
            "val_pr_auc": round(average_precision_score(val["y"], s), 3),
            "FP": int(fp),
            "FN": int(fn),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("reports/baselines_val.csv", index=False)


if __name__ == "__main__":
    main()
