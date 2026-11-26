import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/spam_model.joblib"))
MAX_CHARS = 1000

state = {}


@asynccontextmanager
async def lifespan(app):
    try:
        state["bundle"] = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        pass  # endpoints answer 503 until a model is available
    yield
    state.clear()


app = FastAPI(title="SMS spam classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    score: float  # SVM decision score, NOT a probability
    threshold: float
    model_version: str


def get_bundle():
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


@app.get("/health")
def health():
    bundle = get_bundle()
    return {"status": "ok", "model_version": bundle["version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bundle = get_bundle()
    score = float(bundle["model"].decision_function([req.text])[0])
    label = "spam" if score >= bundle["threshold"] else "ham"
    return PredictResponse(
        label=label,
        score=round(score, 4),
        threshold=round(float(bundle["threshold"]), 4),
        model_version=bundle["version"],
    )
