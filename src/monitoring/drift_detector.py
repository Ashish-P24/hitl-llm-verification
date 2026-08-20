import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.verification.feature_extractor import extract_features
from src.data.database import get_feedback_df, init_db

BASE_DATASET_PATH = "data/processed/verification_dataset.csv"
TRACKED_FEATURES = [
    "question_length",
    "response_length",
    "word_count",
    "uncertainty_count"
]


class DriftDetector:
    """
    Monitors data drift and operational metrics between baseline training distributions
    and runtime/feedback queries.
    """

    def __init__(self, baseline_path: str = BASE_DATASET_PATH):
        self.baseline_path = baseline_path
        self.baseline_df = self._load_baseline()

    def _load_baseline(self) -> pd.DataFrame:
        if os.path.exists(self.baseline_path):
            return pd.read_csv(self.baseline_path)
        return pd.DataFrame(columns=TRACKED_FEATURES)

    def get_baseline_profile(self) -> Dict[str, Dict[str, float]]:
        """Return mean, std, min, max, and median for all tracked features in baseline."""
        profile = {}
        for feat in TRACKED_FEATURES:
            if feat in self.baseline_df.columns:
                series = self.baseline_df[feat].dropna()
                profile[feat] = {
                    "mean": round(float(series.mean()), 2),
                    "std": round(float(series.std()), 2),
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2),
                    "median": round(float(series.median()), 2)
                }
        return profile

    def detect_feature_drift(
        self,
        current_data: Optional[pd.DataFrame] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform 2-sample Kolmogorov-Smirnov statistical test between baseline and current data.
        If current_data is not provided, extracts features from feedback database.
        """
        if current_data is None:
            init_db()
            feedback_df = get_feedback_df()
            if feedback_df.empty or len(feedback_df) < 3:
                return {
                    "status": "INSUFFICIENT_DATA",
                    "message": "At least 3 runtime/feedback samples required to compute drift.",
                    "sample_count": len(feedback_df),
                    "drift_results": {}
                }

            extracted = []
            for _, row in feedback_df.iterrows():
                f = extract_features(row["question"], row["response"])
                extracted.append(f)
            current_data = pd.DataFrame(extracted)

        drift_results = {}
        drift_detected_count = 0

        for feat in TRACKED_FEATURES:
            if feat not in self.baseline_df.columns or feat not in current_data.columns:
                continue

            base_vals = self.baseline_df[feat].dropna().values
            curr_vals = current_data[feat].dropna().values

            if len(curr_vals) < 2 or len(base_vals) < 2:
                continue

            ks_stat, p_value = stats.ks_2samp(base_vals, curr_vals)
            base_mean = float(np.mean(base_vals))
            curr_mean = float(np.mean(curr_vals))
            mean_pct_change = ((curr_mean - base_mean) / (base_mean + 1e-5)) * 100

            has_drift = bool(p_value < alpha)
            if has_drift:
                drift_detected_count += 1

            drift_results[feat] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "has_drift": has_drift,
                "baseline_mean": round(base_mean, 2),
                "current_mean": round(curr_mean, 2),
                "mean_pct_change": round(mean_pct_change, 2)
            }

        return {
            "status": "SUCCESS",
            "alpha": alpha,
            "sample_count": len(current_data),
            "drift_detected": drift_detected_count > 0,
            "drift_features_count": drift_detected_count,
            "drift_results": drift_results
        }

    def get_operational_metrics(self) -> Dict[str, Any]:
        """
        Calculate key system health and HITL operational metrics.
        """
        init_db()
        feedback_df = get_feedback_df()
        total_feedback = len(feedback_df)

        if total_feedback == 0:
            return {
                "total_reviewed_queries": 0,
                "auto_accept_estimate": "N/A",
                "human_review_rate": "N/A",
                "human_agreement_rate": 0.0,
                "rejection_rate": 0.0,
                "edit_rate": 0.0,
                "avg_risk_probability": 0.0
            }

        approved = len(feedback_df[feedback_df["human_decision"] == "APPROVE"])
        rejected = len(feedback_df[feedback_df["human_decision"] == "REJECT"])
        edited = len(feedback_df[feedback_df["human_decision"] == "EDIT"])

        agreement_rate = round(approved / total_feedback, 4)
        rejection_rate = round(rejected / total_feedback, 4)
        edit_rate = round(edited / total_feedback, 4)
        avg_risk = round(float(feedback_df["risk_probability"].mean()), 4)

        return {
            "total_reviewed_queries": total_feedback,
            "approved_count": approved,
            "rejected_count": rejected,
            "edited_count": edited,
            "human_agreement_rate": agreement_rate,
            "rejection_rate": rejection_rate,
            "edit_rate": edit_rate,
            "avg_risk_probability": avg_risk
        }


if __name__ == "__main__":
    detector = DriftDetector()
    print("Baseline Profile:")
    print(detector.get_baseline_profile())
    print("\nOperational Metrics:")
    print(detector.get_operational_metrics())
    print("\nDrift Analysis:")
    print(detector.detect_feature_drift())
