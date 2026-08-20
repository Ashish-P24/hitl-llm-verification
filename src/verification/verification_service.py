from src.llm_service import generate_response
from src.verification.router import VerificationRouter


class VerificationService:

    def __init__(self):
        self.router = VerificationRouter()

    def process_question(self, question: str) -> dict:

        # Step 1: Generate LLM response
        response = generate_response(question)

        # Step 2: Assess risk and route
        routing_result = self.router.route(
            question,
            response
        )

        return {
            "question": question,
            "response": response,
            "risk_probability": routing_result["risk_probability"],
            "decision": routing_result["decision"],
            "features": routing_result["features"]
        }


if __name__ == "__main__":

    service = VerificationService()

    question = "Should I stop taking my prescribed medication?"

    result = service.process_question(question)

    print("\n================ VERIFICATION RESULT ================")

    print("\nQuestion:")
    print(result["question"])

    print("\nLLM Response:")
    print(result["response"])

    print("\nRisk Probability:")
    print(result["risk_probability"])

    print("\nDecision:")
    print(result["decision"])

    print("\nFeatures:")
    print(result["features"])