import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from src.baselines import load
from src.data import SEED
from src.threshold import oof_scores


def word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           min_df=3, sublinear_tf=True, lowercase=False)


def candidates():
    for C in (0.1, 0.3, 1.0, 3.0):
        yield f"word, C={C}", make_pipeline(word_vec(), LinearSVC(C=C, random_state=SEED))
        yield f"char, C={C}", make_pipeline(char_vec(), LinearSVC(C=C, random_state=SEED))
        yield f"word+char, C={C}", make_pipeline(
            FeatureUnion([("w", word_vec()), ("c", char_vec())]),
            LinearSVC(C=C, random_state=SEED))


def main():
    train = load("train")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in candidates():
        oof = oof_scores(model, train["text"], train["y"], skf)
        rows.append({"config": name,
                     "oof_pr_auc": round(average_precision_score(train["y"], oof), 4)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False)
    print()
    print(out.to_string(index=False))
    out.to_csv("reports/tfidf_tuning_oof.csv", index=False)


if __name__ == "__main__":
    main()
