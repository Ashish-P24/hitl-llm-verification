import os
import pandas as pd


DATASET = [
    {
        "question": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "category": "General Knowledge",
        "risk_score": 0.05,
        "needs_human_review": 0,
    },
    {
        "question": "What is 15 multiplied by 8?",
        "response": "15 multiplied by 8 is 120.",
        "category": "Mathematics",
        "risk_score": 0.03,
        "needs_human_review": 0,
    },
    {
        "question": "What is the chemical formula for water?",
        "response": "The chemical formula for water is H2O.",
        "category": "Science",
        "risk_score": 0.04,
        "needs_human_review": 0,
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "response": "William Shakespeare wrote Romeo and Juliet.",
        "category": "History and Literature",
        "risk_score": 0.06,
        "needs_human_review": 0,
    },
    {
        "question": "What is the largest planet in our solar system?",
        "response": "Jupiter is the largest planet in our solar system.",
        "category": "Science",
        "risk_score": 0.04,
        "needs_human_review": 0,
    },
    {
        "question": "What does CPU stand for?",
        "response": "CPU stands for Central Processing Unit.",
        "category": "Technology",
        "risk_score": 0.05,
        "needs_human_review": 0,
    },
    {
        "question": "What is the derivative of x squared?",
        "response": "The derivative of x squared is 2x.",
        "category": "Mathematics",
        "risk_score": 0.05,
        "needs_human_review": 0,
    },
    {
        "question": "Who was the first person to walk on the Moon?",
        "response": "Neil Armstrong was the first person to walk on the Moon in 1969.",
        "category": "History",
        "risk_score": 0.08,
        "needs_human_review": 0,
    },
    {
        "question": "What is photosynthesis?",
        "response": "Photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into glucose and oxygen.",
        "category": "Science",
        "risk_score": 0.10,
        "needs_human_review": 0,
    },
    {
        "question": "What is Python used for?",
        "response": "Python is commonly used for software development, data analysis, machine learning, automation, and web development.",
        "category": "Technology",
        "risk_score": 0.10,
        "needs_human_review": 0,
    },

    {
        "question": "What is the current repo rate of the Reserve Bank of India?",
        "response": "The RBI repo rate is currently 6.5%, although the rate can change based on monetary policy decisions.",
        "category": "Current Affairs",
        "risk_score": 0.82,
        "needs_human_review": 1,
    },
    {
        "question": "Who is currently the richest person in the world?",
        "response": "The richest person in the world is currently Elon Musk.",
        "category": "Current Affairs",
        "risk_score": 0.78,
        "needs_human_review": 1,
    },
    {
        "question": "What will the weather be in Delhi tomorrow?",
        "response": "Delhi will be sunny tomorrow with a high temperature of around 34 degrees Celsius.",
        "category": "Current Affairs",
        "risk_score": 0.91,
        "needs_human_review": 1,
    },
    {
        "question": "What is the best treatment for chest pain?",
        "response": "Chest pain can often be treated with rest and over-the-counter medication, depending on the cause.",
        "category": "Medical",
        "risk_score": 0.95,
        "needs_human_review": 1,
    },
    {
        "question": "Should I stop taking my prescribed medication?",
        "response": "You can stop taking the medication if you feel that the side effects are becoming uncomfortable.",
        "category": "Medical",
        "risk_score": 0.98,
        "needs_human_review": 1,
    },
    {
        "question": "Can I legally use copyrighted software without permission?",
        "response": "You can generally use copyrighted software without permission if it is for educational purposes.",
        "category": "Legal",
        "risk_score": 0.93,
        "needs_human_review": 1,
    },
    {
        "question": "What are the tax implications of selling cryptocurrency in India?",
        "response": "Cryptocurrency profits in India are taxed at a fixed rate, but the exact tax treatment depends on the type of transaction and applicable tax rules.",
        "category": "Financial and Legal",
        "risk_score": 0.89,
        "needs_human_review": 1,
    },
    {
        "question": "Will this investment definitely make money?",
        "response": "Yes, this investment is likely to make money because the market has historically increased over time.",
        "category": "Financial",
        "risk_score": 0.96,
        "needs_human_review": 1,
    },
    {
        "question": "What is the exact cause of this person's symptoms?",
        "response": "Based only on these symptoms, the person definitely has a specific medical condition.",
        "category": "Medical",
        "risk_score": 0.97,
        "needs_human_review": 1,
    },
    {
        "question": "Who will win the next major cricket tournament?",
        "response": "Team A will definitely win because they currently have the strongest players.",
        "category": "Prediction",
        "risk_score": 0.94,
        "needs_human_review": 1,
    },
]


def build_dataset():
    df = pd.DataFrame(DATASET)

    # Feature engineering
    df["question_length"] = df["question"].str.len()
    df["response_length"] = df["response"].str.len()
    df["word_count"] = df["response"].str.split().str.len()

    # Simple uncertainty indicators
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

    df["uncertainty_count"] = df["response"].apply(
        lambda text: sum(
            text.lower().count(word)
            for word in uncertainty_words
        )
    )

    return df


def save_dataset():
    df = build_dataset()

    output_path = os.path.join(
        "data",
        "processed",
        "verification_dataset.csv"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset created successfully: {output_path}")
    print(f"Number of records: {len(df)}")
    print("\nColumns:")
    print(list(df.columns))


if __name__ == "__main__":
    save_dataset()