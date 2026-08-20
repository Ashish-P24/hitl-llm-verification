def extract_features(question: str, response: str) -> dict:
    """
    Extract the same features used during model training.
    """

    question_length = len(question)
    response_length = len(response)
    word_count = len(response.split())

    uncertainty_words = [
        "maybe",
        "might",
        "possibly",
        "likely",
        "probably",
        "could",
        "uncertain",
        "depends",
    ]

    uncertainty_count = sum(
        response.lower().count(word)
        for word in uncertainty_words
    )

    return {
        "question_length": question_length,
        "response_length": response_length,
        "word_count": word_count,
        "uncertainty_count": uncertainty_count,
    }