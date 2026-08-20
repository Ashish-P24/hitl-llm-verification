import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.verification.risk_service import RiskService


class VerificationRouter:

    def __init__(self, review_threshold: float = 0.50):
        self.risk_service = RiskService()
        self.review_threshold = review_threshold

    def route(self, question: str, response: str, threshold: float = None) -> dict:
        curr_threshold = threshold if threshold is not None else self.review_threshold

        assessment = self.risk_service.assess_risk(
            question,
            response
        )

        risk_probability = assessment["risk_probability"]

        if risk_probability >= curr_threshold:
            decision = "HUMAN_REVIEW"
        else:
            decision = "AUTO_ACCEPT"

        return {
            "risk_probability": risk_probability,
            "decision": decision,
            "features": assessment["features"],
            "threshold_used": curr_threshold
        }


if __name__ == "__main__":
    router = VerificationRouter()
    result = router.route(
        "What is the capital of France?",
        "The capital of France is Paris."
    )
    print("\nVerification Routing:")
    print(result)