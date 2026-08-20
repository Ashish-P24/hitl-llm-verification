import os
import joblib
import pandas as pd

from src.verification.feature_extractor import extract_features


MODEL_PATH = "models/risk_model.joblib"


class RiskService:

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Risk model not found: {MODEL_PATH}"
            )

        saved_model = joblib.load(MODEL_PATH)

        self.model = saved_model["model"]
        self.features = saved_model["features"]

    def assess_risk(self, question: str, response: str) -> dict:

        # Extract features from new response
        feature_values = extract_features(
            question,
            response
        )

        # Convert features into DataFrame
        X = pd.DataFrame(
            [feature_values],
            columns=self.features
        )

        # Predict human review requirement
        prediction = self.model.predict(X)[0]

        # Get probability of requiring human review
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