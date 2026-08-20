import unittest
import os
import shutil
import pandas as pd
import numpy as np

from src.verification.feature_extractor import extract_features
from src.verification.risk_service import RiskService
from src.verification.router import VerificationRouter
from src.verification.verification_service import VerificationService
from src.data.database import (
    init_db,
    save_feedback,
    get_all_feedback,
    get_feedback_df,
    get_feedback_stats
)
from src.models.retrain import (
    train_candidate_model,
    deploy_candidate_model,
    rollback_model
)
from src.models.model_comparison import run_model_comparison
from src.monitoring.drift_detector import DriftDetector

TEST_DB_PATH = "data/test_feedback.db"


class TestHITLVerificationPipeline(unittest.TestCase):

    def setUp(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        init_db(TEST_DB_PATH)

    def tearDown(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_feature_extractor(self):
        question = "What is the capital of France?"
        response = "The capital of France is Paris."
        features = extract_features(question, response)

        self.assertIn("question_length", features)
        self.assertIn("response_length", features)
        self.assertIn("word_count", features)
        self.assertIn("uncertainty_count", features)

        self.assertEqual(features["question_length"], len(question))
        self.assertEqual(features["response_length"], len(response))
        self.assertEqual(features["word_count"], 6)
        self.assertEqual(features["uncertainty_count"], 0)

        # Test uncertainty count
        uncertain_resp = "It might be Paris, but maybe it depends on the situation."
        uncertain_f = extract_features(question, uncertain_resp)
        self.assertGreaterEqual(uncertain_f["uncertainty_count"], 3)

    def test_risk_service(self):
        service = RiskService()
        low_risk = service.assess_risk("What is 2+2?", "2+2 equals 4.")
        self.assertIn("risk_probability", low_risk)
        self.assertIn("needs_human_review", low_risk)
        self.assertLess(low_risk["risk_probability"], 0.5)

        high_risk = service.assess_risk(
            "Should I stop my medication?",
            "You might consider reducing your medicine dosage if you feel better, though it possibly depends on the condition."
        )
        self.assertGreaterEqual(high_risk["risk_probability"], 0.5)

    def test_router_and_verification_service(self):
        service = VerificationService(review_threshold=0.5)
        result = service.process_question("What is the capital of France?")
        self.assertIn("decision", result)
        self.assertEqual(result["decision"], "AUTO_ACCEPT")

        med_result = service.process_question("Should I stop taking my prescribed medication?")
        self.assertEqual(med_result["decision"], "HUMAN_REVIEW")

    def test_database_operations(self):
        record_id = save_feedback(
            question="Is crypto guaranteed profit?",
            response="Crypto will maybe 10x your money.",
            risk_probability=0.88,
            model_decision="HUMAN_REVIEW",
            human_decision="REJECT",
            corrected_response="Crypto does not guarantee returns.",
            reviewer_comment="Dangerous investment claim.",
            category="Financial",
            db_path=TEST_DB_PATH
        )
        self.assertEqual(record_id, 1)

        all_records = get_all_feedback(TEST_DB_PATH)
        self.assertEqual(len(all_records), 1)
        self.assertEqual(all_records[0]["human_decision"], "REJECT")

        df = get_feedback_df(TEST_DB_PATH)
        self.assertEqual(len(df), 1)

        stats = get_feedback_stats(TEST_DB_PATH)
        self.assertEqual(stats["total_reviews"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(stats["approval_rate"], 0.0)

    def test_retraining_pipeline(self):
        result = train_candidate_model()
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("candidate_metrics", result)
        self.assertGreaterEqual(result["candidate_metrics"]["accuracy"], 0.80)

    def test_model_comparison(self):
        comp = run_model_comparison()
        self.assertIn("comparison_table", comp)
        self.assertEqual(len(comp["comparison_table"]), 5)
        self.assertIn(comp["best_model"], [
            "Random Forest", "Gradient Boosting", "Logistic Regression",
            "Support Vector Machine", "Decision Tree"
        ])

    def test_drift_detector(self):
        detector = DriftDetector()
        profile = detector.get_baseline_profile()
        self.assertIn("question_length", profile)
        self.assertIn("mean", profile["question_length"])

        op_metrics = detector.get_operational_metrics()
        self.assertIn("total_reviewed_queries", op_metrics)


if __name__ == "__main__":
    unittest.main()
