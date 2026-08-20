import os
from dotenv import load_dotenv

load_dotenv()

# Built-in knowledge base for simulation when API key is unavailable
_MOCK_KNOWLEDGE_BASE = {
    "capital of france": "The capital of France is Paris. It is also the country's most populous city.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s).",
    "moon": "Neil Armstrong was the first person to walk on the Moon on July 20, 1969, during the Apollo 11 mission.",
    "medication": "Medication should not generally be stopped without discussing the decision with an appropriate healthcare professional. Stopping suddenly could cause adverse effects and might depend on your individual clinical condition.",
    "diabetes": "Diabetes may cause increased thirst, frequent urination, fatigue, and unexplained weight loss. Symptoms can vary between individuals and might require professional medical testing.",
    "crypto": "Investing in cryptocurrency carries high market volatility. It could easily generate 10x returns or lose value depending on unpredictable market trends. Returns cannot be guaranteed.",
    "rbi repo rate": "The Reserve Bank of India (RBI) repo rate is subject to periodic monetary policy reviews. It might fluctuate depending on macroeconomic conditions and inflation rates.",
    "contract": "Whether signing is legally permissible depends on the specific agency agreement, power of attorney, and applicable jurisdiction. You cannot generally bind a partner without authorization."
}


def _simulate_llm_response(question: str) -> str:
    """Generate high-fidelity domain-appropriate responses when API key is not configured."""
    q_lower = question.lower().strip()

    for key, answer in _MOCK_KNOWLEDGE_BASE.items():
        if key in q_lower:
            return answer

    # Domain heuristic fallback generator
    if any(w in q_lower for w in ["medicine", "dose", "pain", "doctor", "health", "symptom", "disease"]):
        return (
            "Medical conditions and treatments can be complex and might depend heavily on individual factors. "
            "It is possible that symptoms indicate an underlying issue, but you cannot generally self-diagnose and should consult a licensed physician."
        )
    elif any(w in q_lower for w in ["stock", "invest", "crypto", "profit", "return", "bank", "money"]):
        return (
            "Financial investments carry inherent market risks. Outcomes could vary and might depend on macroeconomic conditions. "
            "Returns cannot be guaranteed."
        )
    elif any(w in q_lower for w in ["court", "law", "sue", "legal", "clause", "contract", "police"]):
        return (
            "Legal rights and liabilities depend strictly on regional statutes and relevant case law. "
            "Specific contractual terms could be interpreted differently in court, so seek formal legal counsel."
        )
    else:
        return f"Regarding '{question}': This is a standard factual topic. The foundational principles are well-documented across literature and empirical records."


def generate_response(question: str) -> str:
    """
    Generate an LLM response using Google Gemini API if configured,
    or use simulated responses for offline testing.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key and api_key != "YOUR_GEMINI_API_KEY" and api_key.strip():
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question
            )
            return response.text
        except Exception as err:
            # If API call fails (e.g. rate limit, quota, network), fallback to simulation
            print(f"[LLM Service Warning] Gemini API call failed ({err}). Falling back to simulation.")
            return _simulate_llm_response(question)

    return _simulate_llm_response(question)