# text-classifier: SMS spam detection with a measured evaluation

[![CI](https://github.com/lakshanindika27-bot/text-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshanindika27-bot/text-classifier/actions)

A small spam classifier built to be evaluated carefully rather than to chase a score: duplicates removed before splitting, a precision target chosen up front, the model selected without touching the test set, and the test set used exactly once. It is served with FastAPI, packaged in Docker and checked by CI.

**Headline result** (final model: character n-gram TF-IDF + linear SVM, test set of 774 messages, 96 spam):

| Metric | Result | 95% bootstrap CI |
|---|---|---|
| Precision | 1.000 (0 false positives) | [1.000, 1.000], degenerate, see note |
| Recall | 0.979 (2 missed spam) | [0.947, 1.000] |
| F1 | 0.989 | [0.973, 1.000] |
| PR-AUC | 0.9954 | not computed |

Confusion matrix: TP 94, FP 0, FN 2, TN 678.

Read this with care. The test set has only 96 spam messages, so the numbers are imprecise. With 0 false positives among 678 ham messages, the "rule of three" only bounds the false-positive rate at roughly 0.4% (3/678) with 95% confidence, and precision at roughly 0.97 or higher (1 - 3/94). The bootstrap interval for precision collapses to [1, 1] because no resample contains a false positive; it does not mean precision is exactly 1. See *Limitations*.

## Evaluation protocol

- **Data:** UCI SMS Spam Collection, 5,574 messages (13.4% spam).
- **Deduplication before splitting:** 415 duplicate messages removed (case and whitespace insensitive), leaving 5,159 (12.4% spam). No message appeared with conflicting labels. Without this step the same text can land in both train and test.
- **Split:** stratified 70/15/15, fixed seed. Train 3,611 (449 spam), validation 774 (97 spam), test 774 (96 spam). A pytest checks that the splits do not overlap.
- **Metrics:** accuracy is not reported as a headline (always predicting "ham" scores about 87%). Precision, recall, F1, PR-AUC and the confusion matrix are used for the spam class.
- **Operating point:** blocking a legitimate message is worse than letting spam through, so the decision threshold is the one that maximises recall subject to **precision >= 0.98 on the training set's out-of-fold predictions** (5-fold). It is never tuned on validation or test.
- **Model selection:** hyperparameters and the final model were chosen using train cross-validation and the validation set. The test set was used once, after those choices were fixed.

## Models compared (validation set, 774 messages, 97 spam)

Threshold column: "default" means the model's default decision rule, "p>=0.98" means the threshold from the protocol above.

| Model | Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| B0 majority class | default | 0.000 | 0.000 | 0.000 | 0 | 97 |
| B1 TF-IDF (word) + LogReg | default | 1.000 | 0.701 | 0.824 | 0 | 29 |
| B1 TF-IDF (word) + LogReg | p>=0.98 | 0.976 | 0.845 | 0.906 | 2 | 15 |
| B2a TF-IDF (word) + Naive Bayes | default | 1.000 | 0.670 | 0.802 | 0 | 32 |
| B2a TF-IDF (word) + Naive Bayes | p>=0.98 | 0.988 | 0.866 | 0.923 | 1 | 13 |
| B2b TF-IDF (word) + linear SVM | default | 0.989 | 0.897 | 0.941 | 1 | 10 |
| B3a nomic-embed-text + LogReg | p>=0.98 | 0.968 | 0.938 | 0.953 | 3 | 6 |
| B3b nomic-embed-text + linear SVM | p>=0.98 | 0.978 | 0.938 | 0.958 | 2 | 6 |
| **B4 TF-IDF (char 2-5) + linear SVM** | p>=0.98 | 0.979 | 0.948 | 0.963 | 2 | 5 |

## Findings

1. **Accuracy hides the problem.** The majority-class baseline looks fine on accuracy and catches no spam.
2. **Choose the threshold on purpose.** Moving LogReg and Naive Bayes from the default threshold to the precision-targeted one raised recall from 0.70 to 0.85 and from 0.67 to 0.87 at a cost of 1-2 extra false positives. The linear SVM was already near that operating point. Comparing models at their default thresholds would have ranked them unfairly.
3. **Tune the simple baseline before reaching for embeddings.** Embeddings (B3) looked clearly better than word-level TF-IDF (B1/B2b). Adding character n-grams to TF-IDF and tuning `C` with cross-validation (out-of-fold PR-AUC rose from about 0.970 to about 0.985) closed the gap: B4 and B3b are indistinguishable (recall 0.948 vs 0.938, FN 5 vs 6, FP 2 vs 2, differences of one message). Embeddings were not worth their cost here.
4. **Choose by deployment cost when accuracy is tied.** B4 needs a single 628 KB `joblib` file and no external service; the embedding models need a running Ollama server at inference time. B4 was selected.

### Error analysis (validation set)

The error review was done on the embedding model (B3b), before the final model was chosen; the final model's validation errors were not inspected separately. B3b made 8 errors (2 false positives, 6 missed spam), reviewed by hand:

- Of the 6 missed spam messages, about 5 were conversational-sounding subscription or chat-line adverts without obvious cues such as prize amounts or "FREE". One ("For sale - arsenal dartboard...") looks like an ordinary classified ad and its label is debatable.
- Both false positives were template-like or chain-style messages, labelled ham, that resemble spam in style.

This is a qualitative observation on 8 messages, not a general claim about the model.

## Limitations

- **Small test set.** 96 spam and 678 ham messages. Confidence intervals are wide, and the validation-to-test improvement (recall 0.948 to 0.979, a difference of about three messages) should be read as noise.
- **Near-duplicates.** Only exact duplicates were removed. Lightly edited copies of template spam (changed numbers or names) may remain on both sides of a split and could make results optimistic.
- **Label noise and age of the data.** The dataset is a 2000s-era collection of English SMS; some labels are ambiguous. Results say little about modern messages, other languages or other kinds of spam.
- **Selection used the validation set.** Several candidates were compared on the same validation set, so validation numbers are mildly optimistic. The test set was used once.
- **The served model was fit on the training set only**, not train + validation.
- **The SVM score is not a probability.** The API returns the raw decision score and the threshold; it does not return `spam_probability`.
- **No monitoring** for drift or abuse, and no authentication or rate limiting on the API. This is a portfolio project, not a production service.
- `models/spam_model.joblib` is a pickle; only load model files from sources you trust.

## API

`POST /predict` returns a label, the SVM decision score, the threshold and the model version. `GET /health` reports the loaded model version. Input is validated (blank text, non-string, or more than 1,000 characters returns HTTP 422); the service returns HTTP 503 if no model is loaded.

    curl -s -X POST localhost:8000/predict \
      -H 'Content-Type: application/json' \
      -d '{"text":"WINNER!! You have won a 1000 prize. Call 09061701461 now"}'
    # {"label":"spam","score":1.328,"threshold":-0.2976,"model_version":"tfidf-char-svm-v1"}

## Run it

Docker (serving only, non-root user, no data in the image):

    docker build -t text-classifier .
    docker run -p 8000:8000 text-classifier

Locally:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    uvicorn src.api:app --port 8000
    python -m pytest

## Reproduce the experiments

Download the dataset from the UCI repository into `data/raw/` (the file `SMSSpamCollection`), then run from the repository root:

    python -m src.data          # dedupe + stratified split
    python -m src.baselines     # B0-B2 on validation
    python -m src.threshold     # precision-targeted thresholds
    python -m src.embed         # needs Ollama + nomic-embed-text
    python -m src.embed_eval    # B3 on validation
    python -m src.tune_tfidf    # word/char n-grams and C, train out-of-fold
    python -m src.compare       # final candidates, same procedure
    python -m src.errors        # validation error analysis (B3b)
    python -m src.final         # fits the final model, writes models/ and
                                # reports/test_results.json (uses the test set; run once)

Result tables are saved under `reports/`.

## Repository layout

| Path | Purpose |
|---|---|
| `src/data.py` | Loading, deduplication, stratified split |
| `src/baselines.py`, `threshold.py`, `embed*.py`, `tune_tfidf.py`, `compare.py`, `errors.py`, `final.py` | Experiments, in the order above |
| `src/api.py` | FastAPI service |
| `tests/` | Split-leakage tests and API tests (12 tests, run in CI) |
| `models/` | Final model artifact |
| `reports/` | Saved experiment results |
| `.github/workflows/ci.yml` | pytest, `pip-audit`, Docker build and smoke test, weekly schedule |

`requirements.txt` holds the serving runtime only (scikit-learn, FastAPI, uvicorn); `requirements-dev.txt` adds the experiment and test tooling.
