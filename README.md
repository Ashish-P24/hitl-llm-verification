# Human-in-the-Loop Adaptive Verification Framework for LLMs

A machine-learning-based verification framework that evaluates the risk of Large Language Model (LLM) responses and intelligently routes potentially risky responses to human reviewers.

The goal is to reduce unnecessary human verification while ensuring that high-risk LLM responses receive human oversight.

---

## 1. Project Overview

Large Language Models such as Gemini, GPT, Llama, and Mistral can generate useful responses across many domains. However, their responses may contain:

- Hallucinations
- Incorrect factual claims
- Overconfident statements
- Unsafe recommendations
- Outdated information
- Financial or legal risks
- Medical risks
- Uncertain predictions

Manually checking every LLM response is expensive and inefficient.

This project implements a **Human-in-the-Loop (HITL) Adaptive Verification Framework**.

Instead of sending every response to a human:

```text
Low-risk response
        ↓
   Automatically accepted
```

while:

```text
High-risk response
        ↓
   Sent for human review
```

The system uses a machine-learning risk classifier to make this routing decision.

---

## 2. Main Objective

The main objective is to develop a system that:

1. Generates an answer using an LLM.
2. Extracts risk-related features from the response.
3. Predicts the probability that human verification is required.
4. Automatically accepts low-risk responses.
5. Routes high-risk responses to a human reviewer.
6. Collects human feedback.
7. Uses feedback for future model improvement and retraining.

---

## 3. Current Project Status

The current implementation is approximately **80–85% complete**.

### Completed

- Project setup & architecture
- Gemini LLM integration with offline simulation fallback
- Synthetic verification dataset (180 samples across 9 categories)
- Feature engineering (length, word count, uncertainty vocabulary)
- Exploratory Data Analysis (`src/data/eda.py`)
- Random Forest risk classifier (`src/models/risk_model.py`)
- Multi-Model Benchmark Comparison (`src/models/model_comparison.py`)
- SQLite Human Feedback Database (`src/data/database.py`)
- Adaptive Learning & Model Retraining Pipeline (`src/models/retrain.py`)
- Statistical Data Drift Detection (KS-Test) & Monitoring (`src/monitoring/drift_detector.py`)
- Verification Router with dynamic thresholds (`src/verification/router.py`)
- End-to-end verification service (`src/verification/verification_service.py`)
- Interactive 5-tab Streamlit web application (`app/streamlit_app.py`)
- Automated Pipeline Test Suite (`tests/test_pipeline.py`)

### Remaining for Final Milestone (15-20%)

- Production containerization (Docker / Cloud Deployment)
- Semantic embedding & RAG-based verification features
- Real-user study and field data collection
- Final project presentation slides and academic report

---

## 4. System Architecture

Current architecture:

```text
                 USER QUESTION
                       |
                       v
                +-------------+
                |   Gemini    |
                |     LLM     |
                +-------------+
                       |
                       v
                 LLM RESPONSE
                       |
                       v
              +------------------+
              | Feature Extractor|
              +------------------+
                       |
                       v
              +------------------+
              |  Random Forest   |
              |   Risk Model     |
              +------------------+
                       |
                       v
                Risk Probability
                       |
                       v
              +------------------+
              | Verification     |
              |     Router       |
              +------------------+
                 /            \
                /              \
               v                v
        AUTO_ACCEPT       HUMAN_REVIEW
```

Final intended architecture:

```text
                       USER
                        |
                        v
                     GEMINI
                        |
                        v
                 GENERATED RESPONSE
                        |
                        v
                RISK ASSESSMENT
                        |
             +----------+----------+
             |                     |
             v                     v
         LOW RISK              HIGH RISK
             |                     |
             v                     v
       AUTO ACCEPT           HUMAN REVIEW
                                   |
                                   v
                           HUMAN DECISION
                                   |
                                   v
                           FEEDBACK STORAGE
                                   |
                                   v
                           FEEDBACK DATASET
                                   |
                                   v
                         MODEL RETRAINING
                                   |
                                   v
                         UPDATED RISK MODEL
```

---

## 5. Technology Stack

**Programming Language**
- Python

**LLM**
- Google Gemini API

**Machine Learning**
- Scikit-learn
- Random Forest Classifier

**Data Processing**
- Pandas
- NumPy

**Model Persistence**
- Joblib

**Environment Configuration**
- python-dotenv

**Planned UI**
- Streamlit

**Planned Storage**
- SQLite or PostgreSQL

**Planned Monitoring**
- Pandas / Streamlit dashboards
- Statistical drift checks
- Model performance tracking

---

## 6. Project Structure

```text
hitl-llm-verification/
│
├── data/
│   └── processed/
│       └── verification_dataset.csv
│
├── models/
│   └── risk_model.joblib
│
├── src/
│   ├── __init__.py
│   │
│   ├── llm_service.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_dataset.py
│   │   └── eda.py
│   │
│   ├── models/
│   │   └── risk_model.py
│   │
│   └── verification/
│       ├── __init__.py
│       ├── feature_extractor.py
│       ├── risk_service.py
│       ├── router.py
│       └── verification_service.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> Note: `.env` and `models/risk_model.joblib` should not be committed to GitHub.

---

## 7. Environment Setup

**Requirements**

- Windows
- Python 3.11+
- VS Code
- Git
- Internet connection
- Gemini API key

---

## 8. Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd hitl-llm-verification
```

---

## 9. Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 10. Install Dependencies

```powershell
pip install -r requirements.txt
```

If a package is missing, install it manually:

```powershell
pip install pandas scikit-learn joblib python-dotenv google-genai
```

---

## 11. Gemini API Configuration

Create a file named `.env` in the project root.

Example:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Do NOT put the actual API key into GitHub.

The `.gitignore` must contain:

```text
.env
```

The application loads the key using:

```python
from dotenv import load_dotenv

load_dotenv()
```

---

## 12. Gemini LLM Service

File: `src/llm_service.py`

This module communicates with Gemini.

The main function is `generate_response(question)`.

- Input: User question
- Output: Gemini-generated response

The LLM is **not** trained by this project. Gemini is used as the response-generation component. The machine-learning model in this project is responsible for **risk classification and routing**.

---

## 13. Dataset

The current dataset contains **180 records** across 9 categories:

```text
General Knowledge
Mathematics
Science
Technology
Medical
Legal
Financial
Current Affairs
Prediction
```

Each category contains approximately 20 examples.

Target: `needs_human_review`

- `0` = No human review required
- `1` = Human review required

Current distribution:

```text
No Review:       121
Human Review:     59
```

---

## 14. Dataset Generation

Handled by `src/data/generate_dataset.py`.

```powershell
python src/data/generate_dataset.py
```

Output saved to `data/processed/verification_dataset.csv`.

---

## 15. Feature Engineering

The system currently extracts four features.

**15.1 Question Length** — number of characters in the question.

```python
question_length = len(question)
```

**15.2 Response Length** — number of characters in the LLM response.

```python
response_length = len(response)
```

**15.3 Word Count** — number of words in the response.

```python
word_count = len(response.split())
```

**15.4 Uncertainty Count** — occurrences of uncertainty-related terms such as: `maybe`, `might`, `possibly`, `likely`, `probably`, `could`, `uncertain`, `depends`.

Example: `"The answer could depend on the situation."` contributes to the uncertainty count.

---

## 16. Exploratory Data Analysis

Implemented in `src/data/eda.py`.

```powershell
python src/data/eda.py
```

The script currently examines:

- Dataset dimensions
- Column information
- Missing values
- Data types
- Target distribution
- Category distribution
- Average features by risk class

Current observations show that responses requiring human review tend to have longer questions, longer responses, and higher uncertainty counts.

---

## 17. Machine Learning Model

Classifier: **Random Forest**

Implemented in `src/models/risk_model.py`.

Uses features: `question_length`, `response_length`, `word_count`, `uncertainty_count`.

The following feature is deliberately **not** used: `risk_score`. Reason: it already represents a manually assigned risk level in the synthetic training data. Using it as an input would cause target leakage and make the evaluation misleading.

---

## 18. Model Training

```powershell
python src/models/risk_model.py
```

The script:

1. Loads the dataset.
2. Selects features.
3. Splits the dataset into training and testing sets.
4. Trains a Random Forest.
5. Evaluates the model.
6. Performs 5-fold stratified cross-validation.
7. Displays feature importance.
8. Saves the trained model.

Saved model: `models/risk_model.joblib` (generated locally, should not be committed to GitHub).

---

## 19. Current Model Results

```text
Dataset:   180 records
Training:  144 samples
Testing:   36 samples
```

- Test accuracy: **94.44%**
- ROC-AUC: **0.9861**

5-fold cross-validation:

```text
Mean Accuracy: 93.33%
Standard Deviation: 4.84%
```

Human-review class:

```text
Precision: 1.00
Recall:    0.83
F1-score:  0.91
```

Confusion matrix:

```text
[[24  0]
 [ 2 10]]
```

---

## 20. Important Model Limitation

The current model should be considered a **baseline risk classifier**, not a complete factual verification model. It does not understand the semantic correctness of an answer.

Example:

```text
Question: What is the capital of France?
Response: The capital of France is London.
```

The current model may not reliably detect that "London" is factually incorrect because it only uses surface-level features. This should be addressed in future development.

Potential future features include:

- Semantic similarity
- Retrieval-based verification
- Factuality checks
- LLM-based verification
- External knowledge sources
- Contradiction detection
- Citation verification
- Domain-specific risk features

---

## 21. Feature Extraction for New Responses

File: `src/verification/feature_extractor.py`

Function: `extract_features(question, response)`

```python
features = extract_features(
    "What is the capital of France?",
    "The capital of France is Paris."
)
```

Output:

```python
{
    "question_length": 30,
    "response_length": 31,
    "word_count": 6,
    "uncertainty_count": 0
}
```

Inference must use the **same** features as model training.

---

## 22. Risk Service

File: `src/verification/risk_service.py`

Loads `models/risk_model.joblib`. Receives a question and response, returns risk probability, human review prediction, and extracted features.

```json
{
    "risk_probability": 0.87,
    "needs_human_review": true,
    "features": {
        "question_length": 45,
        "response_length": 200,
        "word_count": 35,
        "uncertainty_count": 2
    }
}
```

---

## 23. Verification Router

File: `src/verification/router.py`

Current threshold: **0.50**

```text
Risk probability < 0.50   → AUTO_ACCEPT
Risk probability >= 0.50  → HUMAN_REVIEW
```

This threshold should eventually be configurable and optimized based on business costs. For example, in healthcare, false negatives may be significantly more expensive than false positives.

---

## 24. End-to-End Verification Service

File: `src/verification/verification_service.py`

Pipeline:

```text
Question → Gemini → Response → Feature Extraction → Risk Model
→ Risk Probability → Router → AUTO_ACCEPT / HUMAN_REVIEW
```

```powershell
python -m src.verification.verification_service
```

---

## 25. Current Demo

Uses a hardcoded question inside `src/verification/verification_service.py`:

```python
question = "What is the capital of France?"
```

```powershell
python -m src.verification.verification_service
```

Example output:

```text
================ VERIFICATION RESULT ================

Question:
What is the capital of France?

LLM Response:
The capital of France is Paris.

Risk Probability:
0.0

Decision:
AUTO_ACCEPT
```

---

## 26. High-Risk Demo

```python
question = "Should I stop taking my prescribed medication?"
```

```powershell
python -m src.verification.verification_service
```

The system generated a medical response and classified it as:

```text
Risk Probability: 1.0
Decision: HUMAN_REVIEW
```

This demonstrates the second branch of the framework.

---

## 27. Current Demo Flow

```text
1. User submits question
2. Gemini generates response
3. Feature extractor analyzes response
4. Random Forest predicts risk
5. Risk probability is calculated
6. Router makes a decision
7. Low-risk response is automatically accepted
8. High-risk response is sent for human review
```

---

## 28. Remaining Work

### 28.1 Streamlit Web Interface

Recommended structure: `app.py`

The UI should contain:

```text
Question input → Verify button → Gemini response → Risk probability → Decision
```

Example:

```text
HITL LLM Verification

Enter Question:
[....................................]

[ Verify Response ]

LLM Response:
....................................

Risk Probability:
87%

Decision:
HUMAN REVIEW REQUIRED
```

---

## 29. Human Review Interface

When `decision == HUMAN_REVIEW`, the UI should display a review panel.

```text
HUMAN REVIEW REQUIRED

Question:
Should I stop taking my medication?

LLM Response:
...

Risk Probability:
94%

Reviewer Decision:

[ APPROVE ]
[ REJECT ]
[ EDIT RESPONSE ]
```

The reviewer should be able to provide feedback.

---

## 30. Human Feedback Storage

For the prototype, SQLite is recommended because it requires no external database service.

Table: `feedback`

Columns:

```text
id
question
response
risk_probability
model_decision
human_decision
corrected_response
reviewer_comment
timestamp
```

Example row:

```text
1
Question
Gemini Response
0.91
HUMAN_REVIEW
REJECT
Corrected Response
Unsafe recommendation
2026-08-20
```

---

## 31. Adaptive Learning

The human feedback should be used to improve the system.

```text
Human Review → Feedback → Feedback Dataset → Feature Extraction
→ Retraining → New Model → Model Evaluation → Deploy Updated Model
```

The system should **not** automatically replace the production model with every individual human decision. A safer approach:

```text
Collect feedback → Accumulate enough examples → Retrain candidate model
→ Evaluate candidate → Compare against current model → Deploy only if performance improves
```

---

## 32. Monitoring

**Model Monitoring** — track accuracy, precision, recall, F1-score, ROC-AUC, false positives, false negatives.

**Data Drift** — monitor changes in question length, response length, word count, uncertainty count, category distribution, risk probability distribution.

**Business Monitoring** — track percentage auto-accepted, percentage requiring human review, human review workload, review agreement with model, false negative rate, human review time, number of corrected responses.

---

## 33. Ethical Considerations

This system should not be treated as an authority. Human review is especially important for:

- Medical questions
- Legal questions
- Financial decisions
- Safety-critical information
- Personal sensitive situations

The system is designed to assist human decision-making, not replace qualified professionals.

---

## 34. Important Technical Limitation

The current dataset is synthetic. Therefore, 94.44% test accuracy does **not** mean the system will achieve 94.44% accuracy on real-world LLM responses.

The model must eventually be evaluated on:

- Real LLM responses
- More diverse questions
- Larger datasets
- Domain-specific examples
- Adversarial examples
- Real human review decisions

---

## 35. Recommended Future Improvements

**Better Risk Features:** semantic embeddings, sentiment, named entities, medical/legal/financial keywords, question intent, factual claim count, citation presence, contradiction indicators.

**Better Verification:** Retrieval-Augmented Generation, knowledge-base verification, LLM-as-a-judge, external search, fact-checking APIs.

**Better ML:** experiment with Logistic Regression, XGBoost, Gradient Boosting, SVM, neural networks. Compare models using precision, recall, F1, ROC-AUC, PR-AUC. Do not optimize only for accuracy.

---

## 36. Why Human-in-the-Loop?

The main idea of this project is not "AI replaces humans." It is:

```text
AI handles low-risk cases + Humans handle high-risk cases
```

This reduces the amount of manual verification required while preserving human oversight where it matters.

---

## 37. Recommended Development Order

```text
1. Streamlit UI
2. Human Review UI
3. SQLite Feedback Storage
4. Feedback Collection
5. Feedback Dataset
6. Retraining Pipeline
7. Model Comparison
8. Monitoring Dashboard
9. Data Drift Detection
10. Final Testing
11. Documentation
12. Final Demo
```

Do not start with monitoring or advanced ML before the human-review loop works.

---

## 38. Suggested Final Demo

**Case 1 — Low Risk**

```text
Question: What is the capital of France?
Expected: Risk: Low, Decision: AUTO_ACCEPT
```

**Case 2 — Medical Risk**

```text
Question: Should I stop taking my prescribed medication?
Expected: Risk: High, Decision: HUMAN_REVIEW
```

Show the reviewer interface, then approve/reject/edit the response.

**Case 3 — Current or Financial Information**

```text
Question: What is the current RBI repo rate?
or: Will this investment definitely make money?
Expected: HUMAN_REVIEW
```

This demonstrates that the system is designed to route potentially high-risk domains for additional verification.

---

## 39. Git Workflow

```powershell
git pull
git status
git add .
git commit -m "Describe your changes"
git push
```

Recommended commit style:

```text
Add Streamlit verification interface
Add human review workflow
Add feedback database
Add adaptive retraining pipeline
Add monitoring dashboard
```

---

## 40. Security

Never commit `.env`. Never expose `GEMINI_API_KEY`. Never hardcode:

```python
GEMINI_API_KEY = "actual-key"
```

Use:

```python
os.getenv("GEMINI_API_KEY")
```

`.gitignore` should contain:

```text
.env
.venv/
__pycache__/
*.pyc
models/*.joblib
```

---

## 41. Quick Start

### Setup & Installation

```bash
# Clone repository
git clone <REPOSITORY_URL>
cd hitl-llm-verification

# Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\Activate.ps1 # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file (optional; offline simulation works automatically if key is omitted):

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### Execution & Running Modules

1. **Launch Streamlit Web UI (Recommended)**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Train Baseline Risk Model**:
   ```bash
   python src/models/risk_model.py
   ```

3. **Run Multi-Model Benchmark (RF, LogReg, GBDT, SVM, Decision Tree)**:
   ```bash
   python src/models/model_comparison.py
   ```

4. **Initialize & Seed SQLite Feedback Database**:
   ```bash
   python src/data/database.py
   ```

5. **Run Adaptive Retraining Pipeline**:
   ```bash
   python src/models/retrain.py
   ```

6. **Run Statistical Data Drift Detector & Monitoring**:
   ```bash
   python src/monitoring/drift_detector.py
   ```

7. **Run End-to-End Terminal Verification Demo**:
   ```bash
   python src/verification/verification_service.py
   ```

8. **Run Automated Test Suite**:
   ```bash
   python -m unittest tests/test_pipeline.py
   ```

---

## 42. Current Project Position

The framework now covers the complete end-to-end Human-in-the-Loop lifecycle:

```text
1. INFERENCE & ROUTING:
   USER QUESTION → GEMINI / SIMULATOR → LLM RESPONSE
   → FEATURE EXTRACTION → RANDOM FOREST RISK CLASSIFIER
   → PROBABILITY EVALUATION → AUTO_ACCEPT (Low Risk) vs HUMAN_REVIEW (High Risk)

2. HUMAN-IN-THE-LOOP FEEDBACK:
   HUMAN REVIEW STATION → APPROVE / REJECT / EDIT (CORRECTION)
   → SQLITE DATABASE STORAGE (data/feedback.db)

3. ADAPTIVE RETRAINING & CONTINUOUS IMPROVEMENT:
   COMBINED DATASET → CANDIDATE MODEL RETRAINING → 5-FOLD CV EVALUATION
   → BENCHMARK AGAINST PRODUCTION → SAFE DEPLOYMENT & ROLLBACK

4. OBSERVABILITY & DRIFT MONITORING:
   2-SAMPLE KS DRIFT DETECTION → OPERATIONAL KPIS → CSV AUDIT EXPORT
```

---

## 43. Project Goal

The final system provides an intelligent, adaptive verification layer between an LLM and its users.

```text
LLM → Risk Assessment → Intelligent Routing → Human Oversight When Necessary
→ Feedback → Continuous Improvement
```

This architecture balances automation, reliability, human oversight, cost efficiency, safety, and continuous learning.