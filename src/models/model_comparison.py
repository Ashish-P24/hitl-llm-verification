import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = "data/processed/verification_dataset.csv"
FEATURES = [
    "question_length",
    "response_length",
    "word_count",
    "uncertainty_count"
]
TARGET = "needs_human_review"


def run_model_comparison(data_path: str = DATA_PATH) -> Dict[str, Any]:
    """
    Train and evaluate multiple machine learning classifiers to benchmark performance.
    Returns comparison dataframe and best model summary.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
        "Support Vector Machine": SVC(probability=True, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5, class_weight="balanced")
    }

    results: List[Dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Mean CV Acc": round(float(cv_scores.mean()), 4),
            "CV Std": round(float(cv_scores.std()), 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]

    return {
        "comparison_table": results_df.to_dict(orient="records"),
        "comparison_df": results_df,
        "best_model": best_model_name,
        "features_used": FEATURES,
        "dataset_size": len(df)
    }


if __name__ == "__main__":
    print("Running Multi-Model Benchmark Comparison...\n")
    comparison = run_model_comparison()
    print(comparison["comparison_df"].to_string(index=False))
    print(f"\nBest Overall Model: {comparison['best_model']}")
