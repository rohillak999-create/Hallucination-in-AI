"""
Phase 4: FastAPI Prediction Service for Context-Grounded QA Hallucination Detection.

Exposes:
  - GET  /         : Interactive Web UI (`static/index.html`)
  - GET  /health   : Healthcheck & loaded model metadata
  - GET  /examples : Preset QA examples (grounded & hallucinated) for quick testing
  - POST /predict  : Data-contract validated hallucination prediction & grounding signals
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from features import FEATURE_NAMES, extract_grounding_signals

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model" / "hallucination_detector.joblib"
METADATA_PATH = ROOT_DIR / "model" / "model_metadata.json"
STATIC_DIR = ROOT_DIR / "static"


class PredictRequest(BaseModel):
    context: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="Reference context passage against which the answer is verified.",
        examples=[
            "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group. The Oberoi Group is a hotel company with its head office in Delhi."
        ],
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question asked about the reference context.",
        examples=[
            "The Oberoi family is part of a hotel company that has a head office in what city?"
        ],
    )
    answer: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Candidate answer to evaluate for context-grounded hallucination.",
        examples=["The Oberoi family's hotel company is based in Mumbai."],
    )

    @field_validator("context", "question", "answer")
    @classmethod
    def must_not_be_blank(cls, value: str, info: Any) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"'{info.field_name}' must not be blank or whitespace only.")
        return cleaned


class GroundingSignals(BaseModel):
    new_info_grounding_ratio: float
    novel_ungrounded_token_ratio: float
    ungrounded_new_token_count: float
    new_info_substring_in_ctx: float
    bigram_grounding_ratio: float
    best_sentence_support: float
    char_ngram_containment: float
    number_grounding_ratio: float
    ungrounded_number_flag: float
    entity_grounding_ratio: float
    ungrounded_entity_flag: float
    negation_mismatch_flag: float
    contradiction_cue_flag: float


class PredictResponse(BaseModel):
    is_hallucination: bool
    label: str
    hallucination_probability: float
    grounding_score: float
    model_name: str
    grounding_signals: GroundingSignals


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    feature_count: int
    test_metrics: Dict[str, Any]


PRESET_EXAMPLES: List[Dict[str, str]] = [
    {
        "title": "Grounded Answer (Full Sentence)",
        "expected": "grounded",
        "context": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group. The Oberoi Group is a hotel company with its head office in Delhi.",
        "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "answer": "The Oberoi family's hotel company has its head office in Delhi.",
    },
    {
        "title": "Hallucinated Answer (Wrong City Entity)",
        "expected": "hallucinated",
        "context": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group. The Oberoi Group is a hotel company with its head office in Delhi.",
        "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "answer": "The Oberoi family's hotel company is based in Mumbai.",
    },
    {
        "title": "Grounded Answer (Short Extractive Span)",
        "expected": "grounded",
        "context": "Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia. The 6.213 km long track is technically a street circuit where the 2013 Liqui Moly Bathurst 12 Hour was staged.",
        "question": "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?",
        "answer": "6.213 km long",
    },
    {
        "title": "Hallucinated Answer (Fabricated Year / Number)",
        "expected": "hallucinated",
        "context": "House of Anubis is a mystery television series developed for Nickelodeon based on the Dutch-Belgian television series Het Huis Anubis. It first aired in September 2006 and the last episode was broadcast on December 4, 2009.",
        "question": "The Dutch-Belgian television series that House of Anubis was based on first aired in what year?",
        "answer": "The inspiration for House of Anubis first aired in 2003.",
    },
]


def load_model_and_metadata() -> tuple[Any, Dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {MODEL_PATH}. Run `python train.py` first."
        )
    pipeline = joblib.load(MODEL_PATH)
    metadata: Dict[str, Any] = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    return pipeline, metadata


MODEL_PIPELINE, MODEL_METADATA = load_model_and_metadata()

app = FastAPI(
    title="Context-Grounded Hallucination Detector for QA",
    description=(
        "Evaluates whether a candidate answer is faithfully "
        "grounded in the provided reference context or contains hallucinated facts."
    ),
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend UI (static/index.html) not found.")
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_PIPELINE is not None,
        model_name=MODEL_METADATA.get("best_model_name", "HistGradientBoosting_Classifier"),
        feature_count=len(FEATURE_NAMES),
        test_metrics=MODEL_METADATA.get("best_test_metrics", {}),
    )


@app.get("/examples", tags=["Examples"])
def list_examples() -> Dict[str, List[Dict[str, str]]]:
    return {"examples": PRESET_EXAMPLES}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_hallucination(payload: PredictRequest) -> PredictResponse:
    record = [
        {
            "context": payload.context,
            "question": payload.question,
            "answer": payload.answer,
        }
    ]
    pred_class = int(MODEL_PIPELINE.predict(record)[0])
    hallucination_prob = float(MODEL_PIPELINE.predict_proba(record)[0][1])
    grounding_score = round(1.0 - hallucination_prob, 4)
    hallucination_prob_rounded = round(hallucination_prob, 4)

    raw_signals = extract_grounding_signals(
        context=payload.context,
        question=payload.question,
        answer=payload.answer,
    )
    signals = GroundingSignals(
        new_info_grounding_ratio=round(raw_signals["new_info_grounding_ratio"], 4),
        novel_ungrounded_token_ratio=round(raw_signals["novel_ungrounded_token_ratio"], 4),
        ungrounded_new_token_count=round(raw_signals["ungrounded_new_token_count"], 4),
        new_info_substring_in_ctx=round(raw_signals["new_info_substring_in_ctx"], 4),
        bigram_grounding_ratio=round(raw_signals["bigram_grounding_ratio"], 4),
        best_sentence_support=round(raw_signals["best_sentence_support"], 4),
        char_ngram_containment=round(raw_signals["char_ngram_containment"], 4),
        number_grounding_ratio=round(raw_signals["number_grounding_ratio"], 4),
        ungrounded_number_flag=round(raw_signals["ungrounded_number_flag"], 4),
        entity_grounding_ratio=round(raw_signals["entity_grounding_ratio"], 4),
        ungrounded_entity_flag=round(raw_signals["ungrounded_entity_flag"], 4),
        negation_mismatch_flag=round(raw_signals["negation_mismatch_flag"], 4),
        contradiction_cue_flag=round(raw_signals["contradiction_cue_flag"], 4),
    )

    is_hallucination = pred_class == 1
    return PredictResponse(
        is_hallucination=is_hallucination,
        label="hallucinated" if is_hallucination else "grounded",
        hallucination_probability=hallucination_prob_rounded,
        grounding_score=grounding_score,
        model_name=MODEL_METADATA.get("best_model_name", "HistGradientBoosting_Classifier"),
        grounding_signals=signals,
    )
