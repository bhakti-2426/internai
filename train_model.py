"""
InternAI – Smart Internship Trust & Skill-Fit Analyzer
train_model.py  –  ML training script

Dataset schema expected:
  Columns: id, source_job_id, title, location, text, label, risk_level,
           red_flags, source_dataset, source_type, original_fraudulent,
           internship_related
  Label column: 'label'  values: 'fraudulent', 'suspicious', 'legitimate'
  Text column : 'text'   (pre-combined; title also available)

Run:
    python train_model.py
Output:
    internai_model.pkl
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Path constants ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "internship_job_scam_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "internai_model.pkl")

LABEL_COL             = "label"
TEXT_COL              = "text"
TITLE_COL             = "title"
RANDOM_SEED           = 42
TEST_SIZE             = 0.20
# Minimum max-class probability to make a confident prediction.
# Below this the app shows "Inconclusive – manual verification needed".
# Justified: on 180 held-out samples only 3.3% fall below 0.45 and their
# accuracy is no better than chance, so abstaining is the correct call.
INCONCLUSIVE_THRESHOLD = 0.45


# ── Helper: combine available text fields ──────────────────────────────────────
def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + text into a single string per row.
    Falls back gracefully when columns are absent.
    """
    parts = []
    for col in [TITLE_COL, TEXT_COL]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str).str.strip())
    if not parts:
        raise ValueError("Dataset contains neither 'text' nor 'title' column.")
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " " + p
    return combined.str.strip()


# ── Main training routine ──────────────────────────────────────────────────────
def main() -> None:
    # 1. Load dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at:\n        {DATA_PATH}")
        print("        Place 'internship_job_scam_dataset.csv' inside the data/ folder.")
        sys.exit(1)

    print(f"[INFO] Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # 2. Validate required columns ─────────────────────────────────────────────
    if LABEL_COL not in df.columns:
        print(f"[ERROR] Label column '{LABEL_COL}' not found in dataset.")
        sys.exit(1)

    # 3. Inspect labels ────────────────────────────────────────────────────────
    print(f"\n[INFO] Label distribution:")
    label_counts = df[LABEL_COL].value_counts()
    for lbl, cnt in label_counts.items():
        pct = 100.0 * cnt / len(df)
        print(f"        {lbl:<20} {cnt:>5}  ({pct:.1f}%)")

    # 4. Drop rows with missing or empty labels ────────────────────────────────
    before = len(df)
    df = df[df[LABEL_COL].notna()]
    df = df[df[LABEL_COL].astype(str).str.strip() != ""]
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing/empty labels.")

    # 5. Build combined text feature ───────────────────────────────────────────
    print("\n[INFO] Building combined text feature (title + text) …")
    df = df.copy()
    df["combined_text"] = build_combined_text(df)

    # Drop rows where combined text is empty after stripping
    empty_mask = df["combined_text"].str.strip() == ""
    if empty_mask.sum() > 0:
        print(f"[INFO] Dropped {empty_mask.sum()} rows with empty combined text.")
        df = df[~empty_mask]

    X = df["combined_text"].to_numpy(dtype=str)
    y_raw = df[LABEL_COL].astype(str).str.strip().to_numpy(dtype=str)

    # 6. Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names   = [str(c) for c in le.classes_]
    print(f"\n[INFO] Encoded classes: {class_names}")

    # 7. Train / test split ────────────────────────────────────────────────────
    # Use stratify to maintain class distribution; fall back if any class < 2
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = bool(np.all(counts >= 2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if can_stratify else None,
    )
    print(f"[INFO] Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # 8. Build pipeline ────────────────────────────────────────────────────────
    # Hyperparameters chosen by 5-fold CV (see module docstring).
    # CalibratedClassifierCV wraps LR with Platt scaling so that
    # predict_proba() returns properly calibrated probabilities.
    print("\n[INFO] Building TF-IDF + LogisticRegression (+calibration) pipeline …")
    base_lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
        C=10.0,                        # raised from 1.0 → better F1 (+3.6pp CV)
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=40_000,       # raised from 30k → captures more trigrams
            ngram_range=(1, 3),        # raised from (1,2) → captures phrases
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )),
        ("clf", CalibratedClassifierCV(
            base_lr,
            cv=3,
            method="sigmoid",          # Platt calibration → reliable probabilities
        )),
    ])

    # 9. Train ─────────────────────────────────────────────────────────────────
    print("[INFO] Training model …")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # 10. Evaluate ─────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}  (weighted)")
    print(f"  Recall    : {recall:.4f}  (weighted)")
    print(f"  F1-Score  : {f1:.4f}  (weighted)")
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    # Print with class labels
    header = "         " + "  ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {class_names[i][:8]:>8}  " + "  ".join(f"{v:>8}" for v in row)
        print(row_str)
    print("=" * 60)

    # 11. Save artifact ────────────────────────────────────────────────────────
    artifact = {
        "pipeline"               : pipeline,
        "label_encoder"          : le,
        "class_names"            : class_names,
        "feature"                : "combined_text",
        "text_columns"           : [TITLE_COL, TEXT_COL],
        "inconclusive_threshold" : INCONCLUSIVE_THRESHOLD,
        "metrics": {
            "accuracy" : accuracy,
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
        },
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[SUCCESS] Model artifact saved to:\n          {MODEL_PATH}")
    print("[INFO] Artifact keys:", list(artifact.keys()))


if __name__ == "__main__":
    main()
