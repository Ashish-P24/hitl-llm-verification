import os
import sys
from typing import Optional, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm_service import generate_response
from src.verification.router import VerificationRouter


class VerificationService:

    def __init__(self, review_threshold: float = 0.50):
        self.router = VerificationRouter(review_threshold=review_threshold)

    def process_question(
        self,
        question: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        End-to-end verification pipeline:
        1. Generate response from LLM (Gemini or simulated fallback)
        2. Extract features & predict risk with ML model
        3. Route response to AUTO_ACCEPT or HUMAN_REVIEW
        """
        response = generate_response(question)

        routing_result = self.router.route(
            question,
            response,
            threshold=threshold
        )

        return {
            "question": question,
            "response": response,
            "risk_probability": routing_result["risk_probability"],
            "decision": routing_result["decision"],
            "features": routing_result["features"],
            "threshold_used": routing_result.get("threshold_used", self.router.review_threshold)
        }


if __name__ == "__main__":
    service = VerificationService()

    test_questions = [
        "What is the capital of France?",
        "Should I stop taking my prescribed medication?"
    ]

    for q in test_questions:
        result = service.process_question(q)
        print("\n================ VERIFICATION RESULT ================")
        print(f"Question: {result['question']}")
        print(f"Response: {result['response']}")
        print(f"Risk Probability: {result['risk_probability']}")
        print(f"Decision: {result['decision']}")
        print(f"Features: {result['features']}")