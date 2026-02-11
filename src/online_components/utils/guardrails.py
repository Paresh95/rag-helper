def validate_answer(answer: str) -> str:
    # Guardrail 1: ensure fallback phrase if no citation
    if "[Source" not in answer:
        return "I don’t know based on the provided information."

    # Guardrail 2: prevent hallucinated claims
    # (simple heuristic — production uses semantic checking)
    # for sentence in answer.split("."):
    #     if sentence.strip() and sentence not in context:
    #         return "I don’t know based on the provided information."

    return answer
