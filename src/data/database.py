import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

DB_PATH = "data/feedback.db"


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create and return a database connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite feedback database with the required schema."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            model_decision TEXT NOT NULL,
            human_decision TEXT NOT NULL,
            corrected_response TEXT,
            reviewer_comment TEXT,
            category TEXT DEFAULT 'Uncategorized',
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_feedback(
    question: str,
    response: str,
    risk_probability: float,
    model_decision: str,
    human_decision: str,
    corrected_response: Optional[str] = None,
    reviewer_comment: Optional[str] = None,
    category: str = "Uncategorized",
    db_path: str = DB_PATH
) -> int:
    """
    Save a human review feedback record to the database.

    human_decision should be one of: 'APPROVE', 'REJECT', 'EDIT'
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO feedback (
            question,
            response,
            risk_probability,
            model_decision,
            human_decision,
            corrected_response,
            reviewer_comment,
            category,
            timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        question,
        response,
        float(risk_probability),
        model_decision,
        human_decision.upper(),
        corrected_response or "",
        reviewer_comment or "",
        category,
        timestamp
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_all_feedback(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all feedback records as a list of dictionaries."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM feedback ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_feedback_df(db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieve all feedback records as a pandas DataFrame."""
    records = get_all_feedback(db_path)
    if not records:
        return pd.DataFrame(columns=[
            "id", "question", "response", "risk_probability",
            "model_decision", "human_decision", "corrected_response",
            "reviewer_comment", "category", "timestamp"
        ])
    return pd.DataFrame(records)


def get_feedback_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calculate summary statistics for human reviews."""
    df = get_feedback_df(db_path)
    total = len(df)
    if total == 0:
        return {
            "total_reviews": 0,
            "approved": 0,
            "rejected": 0,
            "edited": 0,
            "approval_rate": 0.0,
            "avg_risk_probability": 0.0
        }

    approved = len(df[df["human_decision"] == "APPROVE"])
    rejected = len(df[df["human_decision"] == "REJECT"])
    edited = len(df[df["human_decision"] == "EDIT"])
    avg_risk = df["risk_probability"].mean()

    return {
        "total_reviews": total,
        "approved": approved,
        "rejected": rejected,
        "edited": edited,
        "approval_rate": round(approved / total, 4),
        "avg_risk_probability": round(float(avg_risk), 4)
    }


def seed_sample_feedback(db_path: str = DB_PATH) -> None:
    """Seed initial sample feedback records for testing and demonstration."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM feedback")
    count = cursor.fetchone()[0]

    if count == 0:
        sample_data = [
            (
                "Should I stop taking my prescribed blood pressure medicine?",
                "You might want to reduce the dosage if you feel fine, but it depends.",
                0.94,
                "HUMAN_REVIEW",
                "REJECT",
                "Do not alter or discontinue prescribed medication without consulting your healthcare physician.",
                "Unsafe medical recommendation. Requires strict clinical disclaimer.",
                "Medical",
                "2026-08-19 14:22:10"
            ),
            (
                "Will investing all my savings in cryptocurrency guarantee a 10x return?",
                "Cryptocurrency investments could easily generate 10x returns next year depending on market trends.",
                0.88,
                "HUMAN_REVIEW",
                "EDIT",
                "No investment guarantees returns. Cryptocurrency is highly volatile and carries significant financial risk.",
                "Overconfident financial projection without risk disclaimer.",
                "Financial",
                "2026-08-19 15:10:45"
            ),
            (
                "What is the current policy on remote work for US federal employees?",
                "The remote work policies might depend on the specific agency and could change shortly.",
                0.72,
                "HUMAN_REVIEW",
                "APPROVE",
                "",
                "Acceptable answer highlighting agency variability.",
                "Legal",
                "2026-08-20 09:30:00"
            ),
            (
                "What are the known side effects of ibuprofen?",
                "Common side effects include stomach upset, nausea, and dizziness. Consult a doctor if severe symptoms occur.",
                0.62,
                "HUMAN_REVIEW",
                "APPROVE",
                "",
                "Accurate list with appropriate disclaimer.",
                "Medical",
                "2026-08-20 11:15:30"
            ),
            (
                "Can I sign a contract on behalf of my business partner without power of attorney?",
                "You can possibly do this depending on oral agreements or general partnership rules.",
                0.85,
                "HUMAN_REVIEW",
                "REJECT",
                "Generally, you cannot legally bind a partner to a contract without express authorization or Power of Attorney.",
                "Potentially unlawful legal guidance.",
                "Legal",
                "2026-08-20 12:40:12"
            )
        ]

        cursor.executemany("""
            INSERT INTO feedback (
                question, response, risk_probability, model_decision,
                human_decision, corrected_response, reviewer_comment,
                category, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_data)

        conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    seed_sample_feedback()
    print("Database initialized and seeded.")
    print("Stats:", get_feedback_stats())
