from src.verification.risk_service import RiskService


class VerificationRouter:

    def __init__(self, review_threshold: float = 0.50):
        self.risk_service = RiskService()
        self.review_threshold = review_threshold

    def route(self, question: str, response: str) -> dict:

        assessment = self.risk_service.assess_risk(
            question,
            response
        )

        risk_probability = assessment["risk_probability"]

        if risk_probability >= self.review_threshold:
            decision = "HUMAN_REVIEW"
        else:
            decision = "AUTO_ACCEPT"

        return {
            "risk_probability": risk_probability,
            "decision": decision,
            "features": assessment["features"]
        }


if __name__ == "__main__":

    router = VerificationRouter()

    result = router.route(
        "What is the capital of France?",
        "The capital of France is Paris."
    )

    print("\nVerification Routing:")
    print(result)