import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.baselines import load, scores
from src.data import SEED
from src.embed_eval import make_models
from src.threshold import TARGET_PRECISION, oof_scores, pick_threshold

NAME = "B3b embeddings + linear svm"


def main():
    train, val = load("train"), load("val")
    Xtr = np.load("data/processed/emb_train.npy")
    Xva = np.load("data/processed/emb_val.npy")
    model = make_models()[NAME]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = oof_scores(model, Xtr, train["y"], skf)
    thr = pick_threshold(train["y"].values, oof, TARGET_PRECISION)
    model.fit(Xtr, train["y"])
    s = scores(model, Xva)
    val = val.assign(score=s, pred=(s >= thr).astype(int))
    fp = val[(val.pred == 1) & (val.y == 0)]
    fn = val[(val.pred == 0) & (val.y == 1)]
    lines = [f"{NAME}, threshold {thr:.3f}", "",
             f"FALSE POSITIVES (ham flagged as spam): {len(fp)}"]
    lines += [f"  [{r.score:+.2f}] {r.text[:200]}" for r in fp.itertuples()]
    lines += ["", f"FALSE NEGATIVES (spam missed): {len(fn)}"]
    lines += [f"  [{r.score:+.2f}] {r.text[:200]}" for r in fn.itertuples()]
    text = "\n".join(lines)
    print(text)
    with open("reports/errors_val.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
