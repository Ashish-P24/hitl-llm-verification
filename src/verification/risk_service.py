import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.verification.feature_extractor import extract_features

MODEL_PATH = "models/risk_model.joblib"


class RiskService:

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Risk model not found: {model_path}. Run 'python src/models/risk_model.py' to train it."
            )

        saved_model = joblib.load(model_path)
        self.model = saved_model["model"]
        self.features = saved_model["features"]

    def assess_risk(self, question: str, response: str) -> dict:
        feature_values = extract_features(
            question,
            response
        )

        X = pd.DataFrame(
            [feature_values],
            columns=self.features
        )

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]

        return {
            "risk_probability": round(float(probability), 4),
            "needs_human_review": bool(prediction),
            "features": feature_values
        }


if __name__ == "__main__":
    service = RiskService()

    result = service.assess_risk(
        "What are the symptoms of diabetes?",
        "Diabetes may cause increased thirst, frequent urination, "
        "fatigue, and unexplained weight loss. Symptoms can vary "
        "between individuals."
    )

    print("\nRisk Assessment:")
    print(result)