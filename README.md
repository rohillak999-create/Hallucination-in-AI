# Context-Grounded Hallucination Detector for Question Answering

**Target Python Version:** Python 3.11  

A context-grounded hallucination detection system for Question Answering (QA) that evaluates whether a generated or candidate answer is faithfully supported by the provided reference context or contains hallucinated content.

---

## 1. Problem Statement and Dataset Source

### Problem Statement
Large Language Models (LLMs) used in Question Answering (QA) systems frequently generate fluent responses that contradict or fabricate facts beyond the provided reference context (**context-grounded hallucinations**). The goal of this project is to build, evaluate, and serve a binary classification model that takes a `(context, question, answer)` triplet and predicts whether the candidate `answer` is **grounded (`0`)** in the reference `context` or **hallucinated (`1`)**, along with a calibrated confidence score and interpretable grounding signals.

### Dataset Source
- **Dataset:** **HaluEval QA Benchmark** (*Li et al., "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models", EMNLP 2023*), derived from multi-hop Wikipedia passages in **HotpotQA**.
- **Raw Dataset ([`data/qa_data.json`](data/qa_data.json)):** `10,000` QA records, each containing:
  - `knowledge`: Reference Wikipedia passage(s).
  - `question`: Question about the passage.
  - `right_answer`: Ground-truth factual answer supported by `knowledge`.
  - `hallucinated_answer`: LLM-generated answer containing intrinsic or extrinsic hallucinations.
- **Processed Dataset (`data/hallucination_qa_processed.csv`):**
  - Generated via [`data/prepare_data.py`](data/prepare_data.py).
  - Each raw record is expanded into two binary classification samples (`label = 0` for `grounded`, `label = 1` for `hallucinated`), yielding **20,000 balanced samples** (`10,000` grounded, `10,000` hallucinated; `0` missing values).
  - **Leak-Free Grouped Split:** Uses `GroupShuffleSplit` (80% train / 20% test, `random_state=42`) grouped by `question_id` (`8,000` train groups = `16,000` rows; `2,000` test groups = `4,000` rows) so the same `(context, question)` pair never leaks across train and test sets.
  - **Bias Mitigation in Feature Engineering ([`features.py`](features.py)):** Raw answer length and token count are intentionally excluded from the feature vector so the classifier cannot exploit surface-length differences between short extractive spans and full-sentence answers, focusing strictly on lexical, n-gram, entity, and numerical grounding against `context`.

---

## 2. Data Contract

### Input Schema (`POST /predict`)
| Field | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `context` | `string` | Required, non-empty (`min_length=5`, `max_length=10000`) | Reference knowledge passage against which the answer is verified. |
| `question` | `string` | Required, non-empty (`min_length=3`, `max_length=2000`) | The question being answered. |
| `answer` | `string` | Required, non-empty (`min_length=1`, `max_length=5000`) | Candidate answer to evaluate for hallucination. |

### Output Schema (`POST /predict`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `is_hallucination` | `boolean` | `true` if the answer is predicted to be hallucinated (`label == 1`), `false` if grounded (`label == 0`). |
| `label` | `string` | `"hallucinated"` or `"grounded"`. |
| `hallucination_probability` | `float` | Predicted probability (`0.0` to `1.0`) that the answer is hallucinated. |
| `grounding_score` | `float` | Complement probability (`1.0 - hallucination_probability`) representing factual support. |
| `grounding_signals` | `object` | Key interpretable feature values (`new_info_grounding_ratio`, `novel_ungrounded_token_ratio`, `entity_grounding_ratio`, `number_grounding_ratio`, `char_ngram_containment`). |
| `model_name` | `string` | Name of the active trained model artifact loaded from `model/`. |

### Error Handling & Edge Cases
- Blank or whitespace-only `context`, `question`, or `answer` strings are rejected with HTTP `422 Unprocessable Entity`.
- Numbers, dates, and named entities introduced in `answer` that are absent from both `context` and `question` trigger `ungrounded_number_flag` / `ungrounded_entity_flag` inside `grounding_signals`.

---

## 3. Setup and Run Instructions

### Local Environment Setup (Python 3.11+)
```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install full development, training, MLflow, and testing dependencies
pip install -r requirements.txt
```

### Dataset Preparation & Model Training (with MLflow)
```bash
# 1. Prepare and validate the 20,000-row binary dataset from data/qa_data.json
python data/prepare_data.py

# 2. Train candidate models, log runs to MLflow (mlruns/), and export best model to model/
python train.py

# 3. (Optional) Launch the MLflow Tracking UI
mlflow ui --port 5000
```

### Running the FastAPI Service & Interactive Web UI
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
- **Interactive Web UI:** `http://127.0.0.1:8000`
- **Swagger OpenAPI Docs:** `http://127.0.0.1:8000/docs`
- **Healthcheck Endpoint:** `http://127.0.0.1:8000/health`

### Running Automated Tests
```bash
python -m pytest tests/ -v
```

---

## 4. Model Comparison and Final Model Choice

All candidate models are trained via [`train.py`](train.py) and evaluated on the leak-free held-out test set (`4,000` samples across `2,000` unseen questions) as well as 5 style-invariance sanity checks (short & full-sentence grounded and hallucinated answers).

### Experiment Comparison Table (Held-Out Test Set, $N = 4,000$)

| Model Name | Key Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log Loss | Sanity Checks |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LogisticRegression_L2` | `C=1.0, penalty='l2', solver='lbfgs'` | `0.9025` | `0.9404` | `0.8595` | `0.8981` | `0.9575` | `0.2649` | `5 / 5` |
| `RandomForest_Classifier` | `n_estimators=200, max_depth=8, min_samples_leaf=10` | `0.9008` | `0.9485` | `0.8475` | `0.8952` | `0.9657` | `0.2498` | `5 / 5` |
| `GradientBoosting_Classifier` | `n_estimators=150, lr=0.08, max_depth=4, subsample=0.85` | `0.9048` | `0.9485` | `0.8560` | `0.8999` | `0.9683` | `0.2377` | `5 / 5` |
| **`HistGradientBoosting_Classifier`** *(Selected)* | `max_iter=200, lr=0.06, max_depth=6, l2=1.0` | **`0.9087`** | **`0.9499`** | **`0.8630`** | **`0.9044`** | **`0.9685`** | **`0.2356`** | **`5 / 5`** |

### Model Choice Justification
- **Selected Model:** `HistGradientBoosting_Classifier` (exported to [`model/hallucination_detector.joblib`](model/hallucination_detector.joblib), metadata in [`model/model_metadata.json`](model/model_metadata.json)).
- **Why it won:**
  1. Achieved the highest **Test F1-Score (`0.9044`)**, **Test Accuracy (`90.87%`)**, **Precision (`94.99%`)**, **Recall (`86.30%`)**, and **ROC-AUC (`0.9685`)**, with the lowest **Log Loss (`0.2356`)**.
  2. Passed **5/5 style-invariance sanity checks**, accurately distinguishing grounded vs. hallucinated answers regardless of whether the candidate answer is a 1-word span (`"Delhi"` vs. `"Mumbai"`) or a complete sentence echoing the question (`"The Oberoi family's hotel company has its head office in Delhi."` vs. `"The Oberoi family's hotel company is based in Mumbai."`).

---

## 5. Team Contributions

| Team Member | Roll Number | Role & Project Contributions |
| :--- | :--- | :--- |
| **Kartik Rohilla** | `24CSU319` | **AI Model Training, Evaluation & Selection:** Designed the classification pipeline in `train.py`, trained and compared the four candidate models (`LogisticRegression`, `RandomForest`, `GradientBoosting`, `HistGradientBoosting`), performed hyperparameter tuning and metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC), and exported the final model artifact. |
| **Sezal Grover** | `24CSU293` | **Dataset Preparation & Context-Grounding Feature Engineering:** Processed and validated the 20,000-sample HaluEval QA dataset with leak-free grouped train/test splitting in `data/prepare_data.py`, and engineered the 17 style-invariant grounding features (lexical coverage, character 3-gram overlap, named-entity verification, numerical/year matching, and negation checks) in `features.py`. |
| **Monish Yadav** | `24CSU321` | **Inference API, Interactive Web UI & Model Verification:** Built the FastAPI prediction endpoints and input/output data-contract validation in `app.py`, designed the interactive Web UI (`static/index.html`) for real-time hallucination checking, and implemented the automated test suite in `tests/test_app.py`. |
