"""
Phase 2: Dataset Validation, Expansion, and Grouped Train/Test Split Preparation.

Reads raw HaluEval QA records from `data/qa_data.json`, validates the data contract,
expands each (knowledge, question, right_answer, hallucinated_answer) record into
two binary classification rows (`label=0` grounded, `label=1` hallucinated),
assigns a leak-free group-based 80/20 train/test split by `question_id`, and writes
`data/hallucination_qa_processed.csv`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from features import FEATURE_NAMES, ContextGroundingFeatureExtractor

RAW_JSON_PATH = ROOT_DIR / "data" / "qa_data.json"
PROCESSED_CSV_PATH = ROOT_DIR / "data" / "hallucination_qa_processed.csv"


def prepare_dataset(
    raw_path: Path = RAW_JSON_PATH,
    output_path: Path = PROCESSED_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    raw_records = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("knowledge", "question", "right_answer", "hallucinated_answer"):
                if key not in record or not str(record[key]).strip():
                    raise ValueError(f"Missing or empty '{key}' at line {line_num}")
            raw_records.append(record)

    expanded_rows = []
    for q_idx, rec in enumerate(raw_records):
        context = str(rec["knowledge"]).strip()
        question = str(rec["question"]).strip()
        right_ans = str(rec["right_answer"]).strip()
        hallucinated_ans = str(rec["hallucinated_answer"]).strip()

        expanded_rows.append(
            {
                "question_id": q_idx,
                "context": context,
                "question": question,
                "answer": right_ans,
                "label": 0,
                "label_name": "grounded",
            }
        )
        expanded_rows.append(
            {
                "question_id": q_idx,
                "context": context,
                "question": question,
                "answer": hallucinated_ans,
                "label": 1,
                "label_name": "hallucinated",
            }
        )

    df = pd.DataFrame(expanded_rows)

    # Leak-free grouped split by question_id so the same (context, question) never
    # appears in both train and test sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["question_id"]))

    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def print_dataset_and_feature_summary(df: pd.DataFrame) -> None:
    print("=== Phase 2: Processed Dataset Summary ===")
    print(f"Total rows: {len(df):,}")
    print(f"Unique (context, question) groups: {df['question_id'].nunique():,}")
    print("\nClass distribution:")
    print(df["label_name"].value_counts())
    print("\nSplit distribution:")
    print(pd.crosstab(df["split"], df["label_name"]))

    # Verify group isolation between train and test
    train_groups = set(df.loc[df["split"] == "train", "question_id"])
    test_groups = set(df.loc[df["split"] == "test", "question_id"])
    overlap = len(train_groups & test_groups)
    print(f"\nGroup overlap between train and test: {overlap} (must be 0)")

    # Compute grounding features on a 2,000-row sample to inspect class separation
    sample_df = df.head(2000)
    extractor = ContextGroundingFeatureExtractor()
    X_sample = extractor.fit_transform(sample_df)
    feat_df = pd.DataFrame(X_sample, columns=FEATURE_NAMES)
    feat_df["label_name"] = sample_df["label_name"].values

    print("\n=== Mean Feature Values by Class (first 2,000 rows) ===")
    summary = feat_df.groupby("label_name")[FEATURE_NAMES].mean().T
    print(summary.round(4).to_string())


if __name__ == "__main__":
    processed_df = prepare_dataset()
    print_dataset_and_feature_summary(processed_df)
