"""
Phase 5: Automated Pytest Suite for Context-Grounded QA Hallucination Detector.

Tests:
  1. Healthcheck endpoint (`GET /health`)
  2. Web UI & preset examples endpoints (`GET /`, `GET /examples`)
  3. Feature extraction invariants (`extract_grounding_signals`, `ContextGroundingFeatureExtractor`)
  4. Valid predictions (`POST /predict`) for both grounded and hallucinated answers
  5. Data contract validation (`POST /predict` with blank, missing, or invalid inputs -> HTTP 422)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from app import app
from features import FEATURE_NAMES, ContextGroundingFeatureExtractor, extract_grounding_signals

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == len(FEATURE_NAMES)
    assert "f1_score" in data["test_metrics"]


def test_web_ui_and_examples_endpoints() -> None:
    ui_res = client.get("/")
    assert ui_res.status_code == 200
    assert "Context-Grounded QA Hallucination Detector" in ui_res.text

    ex_res = client.get("/examples")
    assert ex_res.status_code == 200
    examples = ex_res.json().get("examples", [])
    assert len(examples) >= 4
    for item in examples:
        assert {"title", "expected", "context", "question", "answer"} <= set(item.keys())


def test_feature_extractor_style_invariance() -> None:
    context = (
        "The Oberoi family is an Indian family that is famous for its involvement in hotels, "
        "namely through The Oberoi Group. The Oberoi Group is a hotel company with its head office in Delhi."
    )
    question = "The Oberoi family is part of a hotel company that has a head office in what city?"

    short_signals = extract_grounding_signals(context, question, "Delhi")
    sentence_signals = extract_grounding_signals(
        context,
        question,
        "The Oberoi family's hotel company has its head office in Delhi.",
    )
    assert short_signals["new_info_grounding_ratio"] == 1.0
    assert sentence_signals["new_info_grounding_ratio"] == 1.0
    assert short_signals["novel_ungrounded_token_ratio"] == 0.0
    assert sentence_signals["novel_ungrounded_token_ratio"] == 0.0

    extractor = ContextGroundingFeatureExtractor()
    matrix = extractor.fit_transform([(context, question, "Delhi")])
    assert matrix.shape == (1, len(FEATURE_NAMES))


def test_predict_grounded_short_and_full_sentence() -> None:
    context = (
        "The Oberoi family is an Indian family that is famous for its involvement in hotels, "
        "namely through The Oberoi Group. The Oberoi Group is a hotel company with its head office in Delhi."
    )
    question = "The Oberoi family is part of a hotel company that has a head office in what city?"

    for answer in (
        "Delhi",
        "The Oberoi family's hotel company has its head office in Delhi.",
    ):
        response = client.post(
            "/predict",
            json={"context": context, "question": question, "answer": answer},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["is_hallucination"] is False
        assert body["label"] == "grounded"
        assert 0.0 <= body["hallucination_probability"] < 0.5
        assert 0.5 < body["grounding_score"] <= 1.0


def test_predict_hallucinated_entity_and_number() -> None:
    city_payload = {
        "context": "The Oberoi Group is a hotel company with its head office in Delhi.",
        "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "answer": "The Oberoi family's hotel company is based in Mumbai.",
    }
    res_city = client.post("/predict", json=city_payload)
    assert res_city.status_code == 200
    body_city = res_city.json()
    assert body_city["is_hallucination"] is True
    assert body_city["label"] == "hallucinated"
    assert body_city["hallucination_probability"] >= 0.5
    assert body_city["grounding_signals"]["ungrounded_entity_flag"] == 1.0

    year_payload = {
        "context": "House of Anubis is a mystery television series that first aired in September 2006.",
        "question": "What year did House of Anubis first air?",
        "answer": "House of Anubis first aired in 2003.",
    }
    res_year = client.post("/predict", json=year_payload)
    assert res_year.status_code == 200
    body_year = res_year.json()
    assert body_year["is_hallucination"] is True
    assert body_year["label"] == "hallucinated"
    assert body_year["grounding_signals"]["ungrounded_number_flag"] == 1.0


def test_predict_validation_rejects_invalid_payloads() -> None:
    # Whitespace-only answer
    res_blank = client.post(
        "/predict",
        json={
            "context": "Valid reference context passage here.",
            "question": "What is the answer?",
            "answer": "   ",
        },
    )
    assert res_blank.status_code == 422

    # Missing required field
    res_missing = client.post(
        "/predict",
        json={
            "context": "Valid reference context passage here.",
            "question": "What is the answer?",
        },
    )
    assert res_missing.status_code == 422
