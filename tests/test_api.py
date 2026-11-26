from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api

SPAM = ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
        "Text FA to 87121 to receive entry question(std txt rate)T&C's apply")
HAM = "Ok lar... Joking wif u oni..."


@pytest.fixture()
def client():
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_spam_and_ham_smoke(client):
    spam = client.post("/predict", json={"text": SPAM}).json()
    ham = client.post("/predict", json={"text": HAM}).json()
    assert spam["label"] == "spam"
    assert ham["label"] == "ham"
    assert set(spam) == {"label", "score", "threshold", "model_version"}


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   "},
    {"text": "x" * 1001},
    {},
    {"text": 123},
])
def test_invalid_input_is_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", Path("does-not-exist.joblib"))
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={"text": "hi"}).status_code == 503
