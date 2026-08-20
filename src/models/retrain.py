import os
import sys
import shutil
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.verification.feature_extractor import extract_features
from src.data.database import get_feedback_df, init_db

BASE_DATASET_PATH = "data/processed/verification_dataset.csv"
PRODUCTION_MODEL_PATH = "models/risk_model.joblib"
BACKUP_MODEL_PATH = "models/risk_model_backup.joblib"
CANDIDATE_MODEL_PATH = "models/risk_model_candidate.joblib"

FEATURE_COLUMNS = [
    "question_length",
    "response_length",
    "word_count",
    "uncertainty_count"
]


def load_combined_dataset(
    base_path: str = BASE_DATASET_PATH,
    min_feedback_samples: int = 0
) -> Tuple[pd.DataFrame, int, int]:
    """
    Load baseline dataset and combine with human feedback records.
    Returns (combined_df, base_count, feedback_count).
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base dataset not found at {base_path}")

    base_df = pd.read_csv(base_path)
    base_count = len(base_df)

    init_db()
    feedback_df = get_feedback_df()
    feedback_count = len(feedback_df)

    if feedback_count < min_feedback_samples or feedback_count == 0:
        return base_df, base_count, 0

    feedback_rows = []
    for _, row in feedback_df.iterrows():
        features = extract_features(row["question"], row["response"])
        
        decision = str(row["human_decision"]).upper()
        if decision in ["REJECT", "EDIT"]:
            target_label = 1
        else:
            target_label = 0

        feedback_rows.append({
            "question": row["question"],
            "response": row["response"],
            "category": row.get("category", "Feedback"),
            "risk_score": row["risk_probability"],
            "needs_human_review": target_label,
            "question_length": features["question_length"],
            "response_length": features["response_length"],
            "word_count": features["word_count"],
            "uncertainty_count": features["uncertainty_count"]
        })

    feedback_processed_df = pd.DataFrame(feedback_rows)
    combined_df = pd.concat([base_df, feedback_processed_df], ignore_index=True)
    return combined_df, base_count, feedback_count


def evaluate_model_on_data(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
    else:
        metrics["roc_auc"] = 1.0

    return metrics


def train_candidate_model() -> Dict[str, Any]:
    """
    Train a candidate risk model using combined baseline and human feedback data,
    evaluating performance and comparing against the current production model.
    """
    combined_df, base_count, feedback_count = load_combined_dataset()

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df["needs_human_review"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    candidate_model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
        max_depth=6
    )
    candidate_model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(candidate_model, X, y, cv=cv, scoring="accuracy")

    candidate_metrics = evaluate_model_on_data(candidate_model, X_test, y_test)
    candidate_metrics["cv_mean_accuracy"] = round(float(cv_scores.mean()), 4)
    candidate_metrics["cv_std"] = round(float(cv_scores.std()), 4)

    production_metrics = None
    if os.path.exists(PRODUCTION_MODEL_PATH):
        try:
            prod_saved = joblib.load(PRODUCTION_MODEL_PATH)
            prod_model = prod_saved["model"]
            production_metrics = evaluate_model_on_data(prod_model, X_test, y_test)
        except Exception as e:
            print(f"[Warning] Could not evaluate production model: {e}")

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": candidate_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    os.makedirs(os.path.dirname(CANDIDATE_MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": candidate_model,
        "features": FEATURE_COLUMNS,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_size": len(combined_df),
        "feedback_samples_used": feedback_count,
        "metrics": candidate_metrics
    }, CANDIDATE_MODEL_PATH)

    meets_deployment_threshold = (
        candidate_metrics["accuracy"] >= 0.85 and
        candidate_metrics["recall"] >= 0.70
    )

    return {
        "status": "SUCCESS",
        "dataset_stats": {
            "total_samples": len(combined_df),
            "baseline_samples": base_count,
            "feedback_samples": feedback_count,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
        "feature_importance": importance_df.to_dict(orient="records"),
        "meets_deployment_threshold": meets_deployment_threshold,
        "candidate_path": CANDIDATE_MODEL_PATH
    }


def deploy_candidate_model() -> Dict[str, Any]:
    """Safely deploy candidate model to production."""
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        raise FileNotFoundError("Candidate model not found. Run train_candidate_model() first.")

    os.makedirs(os.path.dirname(PRODUCTION_MODEL_PATH), exist_ok=True)

    if os.path.exists(PRODUCTION_MODEL_PATH):
        shutil.copy2(PRODUCTION_MODEL_PATH, BACKUP_MODEL_PATH)

    shutil.copy2(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)

    return {
        "status": "DEPLOYED",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "production_path": PRODUCTION_MODEL_PATH,
        "backup_path": BACKUP_MODEL_PATH
    }


def rollback_model() -> Dict[str, Any]:
    """Rollback production model to backup version."""
    if not os.path.exists(BACKUP_MODEL_PATH):
        raise FileNotFoundError("No backup model found to rollback to.")

    shutil.copy2(BACKUP_MODEL_PATH, PRODUCTION_MODEL_PATH)
    return {
        "status": "ROLLED_BACK",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "production_path": PRODUCTION_MODEL_PATH
    }


if __name__ == "__main__":
    print("Running adaptive retraining pipeline...")
    result = train_candidate_model()
    print("\n--- Retraining Results ---")
    print(f"Dataset Size: {result['dataset_stats']['total_samples']} (Feedback used: {result['dataset_stats']['feedback_samples']})")
    print(f"Candidate Test Accuracy: {result['candidate_metrics']['accuracy']:.4f}")
    print(f"Candidate Test ROC-AUC:  {result['candidate_metrics']['roc_auc']:.4f}")
    print(f"Candidate Test Recall:   {result['candidate_metrics']['recall']:.4f}")
    print(f"Meets Deployment Threshold: {result['meets_deployment_threshold']}")
