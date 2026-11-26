import hashlib
import json
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

from src.track import EXPERIMENT, TRACKING_URI

NAME = "sms-spam-classifier"
BUNDLE = Path("models/spam_model.joblib")
TEST_RESULTS = Path("reports/test_results.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    digest = sha256(BUNDLE)

    for v in client.search_model_versions(f"name='{NAME}'"):
        if v.tags.get("sha256") == digest:
            print(f"already registered as version {v.version}; nothing to do")
            return

    bundle = joblib.load(BUNDLE)  # trusted: produced by src/final.py
    result = json.loads(TEST_RESULTS.read_text())
    clf = bundle["model"][-1]

    with mlflow.start_run(run_name=f"final: {bundle['version']}"):
        mlflow.set_tags({"stage": "final", "test_set_used": "once"})
        mlflow.log_params({
            "model_version": bundle["version"],
            "classifier": type(clf).__name__,
            "C": clf.C,
            "threshold": round(float(bundle["threshold"]), 4),
            "target_precision": bundle["target_precision"],
            "sklearn_version": bundle["sklearn_version"],
        })
        # Logged from the saved file; the test set is NOT re-evaluated here.
        mlflow.log_metrics({
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "test_f1": result["f1"],
            "test_pr_auc": result["pr_auc"],
            "test_fp": result["FP"],
            "test_fn": result["FN"],
            "n_test": result["n_test"],
            "n_spam": result["n_spam"],
        })
        mlflow.log_artifact(str(TEST_RESULTS))
        mlflow.log_artifact(str(BUNDLE), artifact_path="bundle")
        mlflow.sklearn.log_model(
            bundle["model"], name="model", registered_model_name=NAME)

    versions = client.search_model_versions(f"name='{NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_model_version_tag(NAME, latest.version, "sha256", digest)
    client.set_model_version_tag(
        NAME, latest.version, "threshold", str(round(float(bundle["threshold"]), 4)))
    client.set_registered_model_alias(NAME, "champion", latest.version)
    print(f"registered {NAME} version {latest.version}, alias 'champion'")


if __name__ == "__main__":
    main()
