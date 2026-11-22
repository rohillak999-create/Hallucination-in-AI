"""
Phase 3: Model Training, Evaluation, and MLflow Experiment Tracking.

Trains and evaluates multiple candidate classifiers on the processed HaluEval QA
dataset (`data/hallucination_qa_processed.csv`), logs all parameters, metrics,
and artifacts to MLflow (`mlruns/`), verifies behavior on both benchmark and
style-invariant sanity checks, and exports the best model pipeline to
`model/hallucination_detector.joblib` along with `model/model_metadata.json`.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.prepare_data import PROCESSED_CSV_PATH, prepare_dataset
from features import (
    CAP_ENTITY_RE,
    FEATURE_NAMES,
    NUMBER_RE,
    STOPWORDS,
    ContextGroundingFeatureExtractor,
    tokenize,
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_ARTIFACT_PATH = MODEL_DIR / "hallucination_detector.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MLFLOW_TRACKING_DIR = ROOT_DIR / "mlruns"
EXPERIMENT_NAME = "hallucination-detector-qa"


def build_style_balanced_training_records(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the training split (never the test split) with style-balanced
    counterparts so the classifier does not overfit to surface answer length:
      - Full-sentence grounded answers synthesized from (question + right_answer)
      - Short-span hallucinated answers extracted from ungrounded entities/numbers
        in hallucinated_answer
    """
    augmented_rows: List[Dict[str, Any]] = []

    grounded_df = train_df[train_df["label"] == 0]
    hallucinated_df = train_df[train_df["label"] == 1]

    # 1. Add full-sentence grounded answers (combining question frame + grounded answer)
    for row in grounded_df.itertuples(index=False):
        q_clean = str(row.question).rstrip(" ?.")
        q_words = [w for w in q_clean.split() if w.lower() not in {"what", "which", "who", "where", "when", "how"}]
        prefix = " ".join(q_words[-6:]) if q_words else "The answer"
        full_sentence_grounded = f"{prefix} is {row.answer}."
        augmented_rows.append(
            {
                "question_id": row.question_id,
                "context": row.context,
                "question": row.question,
                "answer": full_sentence_grounded,
                "label": 0,
                "label_name": "grounded",
                "split": "train",
            }
        )

    # 2. Add short-span hallucinated answers (extracting novel ungrounded entities/tokens)
    for row in hallucinated_df.itertuples(index=False):
        ctx_lower = str(row.context).lower()
        q_lower = str(row.question).lower()
        ans_str = str(row.answer)

        # Look for capitalized entities or numbers in hallucinated answer absent from context
        ents = [e for e in CAP_ENTITY_RE.findall(ans_str) if e.lower() not in ctx_lower and e.lower() not in STOPWORDS]
        nums = [n for n in NUMBER_RE.findall(ans_str) if n not in ctx_lower]
        if ents:
            short_hallucinated = " ".join(ents[:2])
        elif nums:
            short_hallucinated = nums[0]
        else:
            novel_toks = [
                t for t in tokenize(ans_str)
                if t not in STOPWORDS and t not in ctx_lower and t not in q_lower
            ]
            if not novel_toks:
                continue
            short_hallucinated = " ".join(novel_toks[:2])

        augmented_rows.append(
            {
                "question_id": row.question_id,
                "context": row.context,
                "question": row.question,
                "answer": short_hallucinated,
                "label": 1,
                "label_name": "hallucinated",
                "split": "train",
            }
        )

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([train_df, aug_df], ignore_index=True)
    return combined.sample(frac=1.0, random_state=42).reset_index(drop=True)


def get_candidate_models() -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Return candidate classifiers and their logged hyperparameter dictionaries."""
    return [
        (
            "LogisticRegression_L2",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "model_family": "LogisticRegression",
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "scaled": True,
            },
        ),
        (
            "RandomForest_Classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "model_family": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
            },
        ),
        (
            "GradientBoosting_Classifier",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                min_samples_leaf=15,
                subsample=0.85,
                random_state=42,
            ),
            {
                "model_family": "GradientBoostingClassifier",
                "n_estimators": 150,
                "learning_rate": 0.08,
                "max_depth": 4,
                "min_samples_leaf": 15,
                "subsample": 0.85,
            },
        ),
        (
            "HistGradientBoosting_Classifier",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.06,
                max_depth=6,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=42,
            ),
            {
                "model_family": "HistGradientBoostingClassifier",
                "max_iter": 200,
                "learning_rate": 0.06,
                "max_depth": 6,
                "min_samples_leaf": 25,
                "l2_regularization": 1.0,
            },
        ),
    ]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def run_sanity_checks(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """Verify predictions on both short and full-sentence grounded & hallucinated answers."""
    cases = [
        {
            "case": "Grounded short span",
            "context": "The Oberoi Group is a hotel company with its head office in Delhi.",
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "answer": "Delhi",
            "expected_label": 0,
        },
        {
            "case": "Grounded full sentence",
            "context": "The Oberoi Group is a hotel company with its head office in Delhi.",
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "answer": "The Oberoi family's hotel company has its head office in Delhi.",
            "expected_label": 0,
        },
        {
            "case": "Hallucinated full sentence",
            "context": "The Oberoi Group is a hotel company with its head office in Delhi.",
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "answer": "The Oberoi family's hotel company is based in Mumbai.",
            "expected_label": 1,
        },
        {
            "case": "Hallucinated short span",
            "context": "The Oberoi Group is a hotel company with its head office in Delhi.",
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "answer": "Mumbai",
            "expected_label": 1,
        },
        {
            "case": "Hallucinated year mismatch",
            "context": "House of Anubis is a mystery television series that first aired in September 2006.",
            "question": "What year did House of Anubis first air?",
            "answer": "House of Anubis first aired in 2003.",
            "expected_label": 1,
        },
    ]
    preds = pipeline.predict(cases)
    probs = pipeline.predict_proba(cases)[:, 1]
    results = []
    for c, pred, prob in zip(cases, preds, probs):
        results.append(
            {
                "case": c["case"],
                "expected": c["expected_label"],
                "predicted": int(pred),
                "hallucination_prob": round(float(prob), 4),
                "passed": int(pred) == c["expected_label"],
            }
        )
    return results


def train_and_track() -> Dict[str, Any]:
    if not PROCESSED_CSV_PATH.exists():
        print("Processed dataset not found; running Phase 2 dataset preparation first...")
        df = prepare_dataset()
    else:
        df = pd.read_csv(PROCESSED_CSV_PATH)

    raw_train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Loaded dataset: {len(raw_train_df):,} raw train rows, {len(test_df):,} held-out test rows.")
    train_df = build_style_balanced_training_records(raw_train_df)
    print(f"Style-balanced training rows (train + style counterparts): {len(train_df):,}")

    # Pre-extract feature matrices once for fast multi-model training
    print("Extracting context-grounding features...")
    t0 = time.time()
    feature_extractor = ContextGroundingFeatureExtractor()
    X_train = feature_extractor.fit_transform(train_df)
    y_train = train_df["label"].to_numpy(dtype=np.int64)

    X_test = feature_extractor.transform(test_df)
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    feat_time = round(time.time() - t0, 2)
    print(f"Feature extraction completed in {feat_time}s (shape: train={X_train.shape}, test={X_test.shape}).")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_runs_summary: List[Dict[str, Any]] = []
    best_run_info: Dict[str, Any] = {}
    best_full_pipeline: Pipeline | None = None
    best_score = (-1.0, -1.0)  # (sanity_pass_rate, test_f1)

    for model_name, estimator, params in get_candidate_models():
        print(f"\n--- Training {model_name} ---")
        run_start = time.time()
        with mlflow.start_run(run_name=model_name) as run:
            estimator.fit(X_train, y_train)
            train_time = round(time.time() - run_start, 3)

            y_train_pred = estimator.predict(X_train)
            y_train_prob = estimator.predict_proba(X_train)[:, 1]
            train_metrics = evaluate_predictions(y_train, y_train_pred, y_train_prob)

            y_test_pred = estimator.predict(X_test)
            y_test_prob = estimator.predict_proba(X_test)[:, 1]
            test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

            full_pipeline = Pipeline(
                [
                    ("features", ContextGroundingFeatureExtractor()),
                    ("model", estimator),
                ]
            )
            sanity_results = run_sanity_checks(full_pipeline)
            sanity_pass_count = sum(1 for r in sanity_results if r["passed"])
            sanity_pass_rate = sanity_pass_count / len(sanity_results)

            # Log parameters
            mlflow.log_params(
                {
                    **params,
                    "num_features": len(FEATURE_NAMES),
                    "raw_train_rows": len(raw_train_df),
                    "augmented_train_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1_score": test_metrics["f1_score"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_log_loss": test_metrics["log_loss"],
                    "train_f1_score": train_metrics["f1_score"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "sanity_pass_rate": sanity_pass_rate,
                    "train_time_sec": train_time,
                }
            )

            # Log full end-to-end scikit-learn pipeline to MLflow
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                artifact_path="model",
            )

            run_record = {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "params": params,
                "test_metrics": test_metrics,
                "train_metrics": train_metrics,
                "sanity_checks": sanity_results,
                "sanity_pass_rate": sanity_pass_rate,
                "train_time_sec": train_time,
            }
            all_runs_summary.append(run_record)

            print(
                f"{model_name}: Test Accuracy={test_metrics['accuracy']:.4f}, "
                f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, "
                f"F1={test_metrics['f1_score']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}, "
                f"Sanity={sanity_pass_count}/{len(sanity_results)}"
            )

            candidate_key = (sanity_pass_rate, test_metrics["f1_score"], test_metrics["roc_auc"])
            if candidate_key > (*best_score, best_run_info.get("test_metrics", {}).get("roc_auc", -1.0)):
                best_score = (sanity_pass_rate, test_metrics["f1_score"])
                best_run_info = run_record
                best_full_pipeline = full_pipeline

    assert best_full_pipeline is not None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_full_pipeline, MODEL_ARTIFACT_PATH)

    metadata = {
        "project": "Context-Grounded Hallucination Detector for QA",
        "experiment_name": EXPERIMENT_NAME,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model_name": best_run_info["model_name"],
        "best_run_id": best_run_info["run_id"],
        "model_artifact_path": str(MODEL_ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "feature_names": FEATURE_NAMES,
        "best_test_metrics": best_run_info["test_metrics"],
        "sanity_checks": best_run_info["sanity_checks"],
        "all_runs": all_runs_summary,
    }
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Phase 3: Best Model Selected ===")
    print(f"Winner: {best_run_info['model_name']} (MLflow run_id={best_run_info['run_id']})")
    print(f"Saved model artifact to: {MODEL_ARTIFACT_PATH}")
    print(f"Saved metadata to:       {MODEL_METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    train_and_track()
