import os
import sys
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np

from src.verification.verification_service import VerificationService
from src.data.database import (
    init_db,
    save_feedback,
    get_all_feedback,
    get_feedback_df,
    get_feedback_stats,
    seed_sample_feedback
)
from src.models.retrain import (
    train_candidate_model,
    deploy_candidate_model,
    rollback_model,
    PRODUCTION_MODEL_PATH
)
from src.models.model_comparison import run_model_comparison
from src.monitoring.drift_detector import DriftDetector

# Page configuration
st.set_page_config(
    page_title="HITL LLM Verification Framework",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-auto-accept {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }
    .badge-human-review {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #EF9A9A;
    }
    .review-box {
        background-color: #FFF8E1;
        border: 1px solid #FFE082;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("HITL Verification")
    st.caption("Adaptive Human-in-the-Loop Framework for LLM Quality & Safety")
    
    st.divider()
    st.subheader("⚙️ Verification Settings")
    threshold = st.slider(
        "Human Review Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Responses with risk probability >= this threshold are routed to human review."
    )
    
    st.divider()
    st.subheader("LLM Backend")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "YOUR_GEMINI_API_KEY" and api_key.strip():
        st.success("Gemini API Active")
    else:
        st.info("Local Simulator Active (No Gemini API Key)")
    
    st.divider()
    st.subheader("🗄️ Feedback Database")
    stats = get_feedback_stats()
    st.write(f"**Total Reviews:** {stats['total_reviews']}")
    st.write(f"**Approval Rate:** {stats['approval_rate'] * 100:.1f}%")
    
    if st.button("Seed Sample Feedback"):
        seed_sample_feedback()
        st.success("Sample reviews added!")
        st.rerun()

# ---------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------
st.title("Human-in-the-Loop Adaptive Verification Framework")
st.write(
    "Intelligently evaluates the risk of Large Language Model responses and routes "
    "potentially risky responses to human reviewers to ensure safety and factual correctness."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Verification Playground",
    "✍️ Human Review Station",
    "Adaptive Retraining",
    "Monitoring & Data Drift",
    "Model Comparison"
])

# ---------------------------------------------------------
# TAB 1: VERIFICATION PLAYGROUND
# ---------------------------------------------------------
with tab1:
    st.header("Interactive LLM Risk Assessment & Routing")
    st.write("Submit a question to generate an LLM response, extract risk features, and route based on the ML classifier.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Enter Question")
        
        # Presets
        preset = st.selectbox(
            "Quick Preset Scenarios:",
            [
                "Custom Question",
                "Low Risk (General): What is the capital of France?",
                "Low Risk (Science): What is the speed of light?",
                "High Risk (Medical): Should I stop taking my prescribed blood pressure medicine?",
                "High Risk (Financial): Will investing all my savings in cryptocurrency guarantee a 10x return?",
                "High Risk (Legal): Can I sign a contract on behalf of my partner without power of attorney?"
            ]
        )

        if preset == "Custom Question":
            default_q = ""
        else:
            default_q = preset.split(": ", 1)[1]

        user_question = st.text_area(
            "Question text:",
            value=default_q,
            height=110,
            placeholder="Type any prompt or question here..."
        )

        category = st.selectbox(
            "Domain Category:",
            ["General Knowledge", "Medical", "Financial", "Legal", "Science", "Technology", "Current Affairs"]
        )

        verify_button = st.button("Generate & Verify Response", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Verification Result")
        
        if verify_button:
            if not user_question.strip():
                st.warning("Please enter a question to assess.")
            else:
                with st.spinner("Generating LLM response and predicting risk..."):
                    try:
                        service = VerificationService(review_threshold=threshold)
                        result = service.process_question(user_question, threshold=threshold)
                        
                        # Store in session state for easy handoff to Review tab
                        st.session_state["last_result"] = result
                        st.session_state["last_category"] = category
                        
                    except Exception as e:
                        st.error(f"Error during verification: {e}")

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            risk_prob = res["risk_probability"]
            decision = res["decision"]
            features = res["features"]

            # Decision Badge
            if decision == "AUTO_ACCEPT":
                st.markdown(
                    f'<div class="badge-auto-accept">✅ AUTO_ACCEPT (Risk: {risk_prob*100:.1f}%)</div>',
                    unsafe_allow_html=True
                )
                st.success("Response is within safe confidence thresholds and automatically accepted.")
            else:
                st.markdown(
                    f'<div class="badge-human-review">⚠️ HUMAN_REVIEW REQUIRED (Risk: {risk_prob*100:.1f}%)</div>',
                    unsafe_allow_html=True
                )
                st.error("High risk detected! This response must be reviewed by a human domain expert.")

            # Progress Bar for Risk
            st.write(f"**Predicted Risk Probability:** `{risk_prob:.4f}` (Threshold: `{threshold:.2f}`)")
            st.progress(float(min(max(risk_prob, 0.0), 1.0)))

            # LLM Response Content
            st.markdown("**LLM Generated Response:**")
            st.info(res["response"])

            # Feature Inspection Cards
            st.markdown("**Extracted Risk Features:**")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Question Length", f"{features['question_length']} chars")
            fc2.metric("Response Length", f"{features['response_length']} chars")
            fc3.metric("Word Count", f"{features['word_count']} words")
            fc4.metric("Uncertainty Terms", f"{features['uncertainty_count']} found")

            # Quick Review Handoff
            if decision == "HUMAN_REVIEW":
                st.markdown("""
                > **Next Step:** Switch to the **'Human Review Station'** tab above to approve, reject, or edit this response and save reviewer feedback.
                """)

# ---------------------------------------------------------
# TAB 2: HUMAN REVIEW STATION
# ---------------------------------------------------------
with tab2:
    st.header("✍️ Human-in-the-Loop Review Station")
    st.write("Review flagged high-risk responses, submit expert decisions, and log feedback for continuous adaptive learning.")

    # Check if there is an active item from Tab 1 or manual entry
    active_q = ""
    active_r = ""
    active_risk = 0.85
    active_cat = "Medical"

    if "last_result" in st.session_state:
        active_q = st.session_state["last_result"]["question"]
        active_r = st.session_state["last_result"]["response"]
        active_risk = st.session_state["last_result"]["risk_probability"]
        active_cat = st.session_state.get("last_category", "Medical")

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.subheader("Flagged Query & Response")
        review_q = st.text_area("Question under Review:", value=active_q, height=90)
        review_r = st.text_area("LLM Output:", value=active_r, height=130)
        review_risk = st.number_input("Assessed Risk Score:", value=float(active_risk), min_value=0.0, max_value=1.0, step=0.01)
        review_category = st.selectbox("Category:", ["Medical", "Financial", "Legal", "General Knowledge", "Science", "Technology", "Other"], index=0)

    with col_r2:
        st.subheader("Human Reviewer Action")
        
        decision_choice = st.radio(
            "Reviewer Verdict:",
            [
                "APPROVE (Output is safe, factual, and verified)",
                "REJECT (Output is harmful, incorrect, or halluncinated)",
                "EDIT (Output requires factual correction or clinical/legal disclaimer)"
            ],
            index=1
        )

        decision_code = "APPROVE" if "APPROVE" in decision_choice else ("REJECT" if "REJECT" in decision_choice else "EDIT")

        corrected_text = ""
        if decision_code == "EDIT":
            corrected_text = st.text_area(
                "Provide Corrected Expert Response:",
                value=review_r,
                height=100,
                placeholder="Write the accurate / safe version of the response..."
            )

        comment_preset = st.multiselect(
            "Review Tags / Reasons:",
            [
                "Unsafe medical guidance",
                "Unverified financial claim",
                "Missing risk disclaimers",
                "Factual hallucination",
                "Overconfident assertions",
                "Sound and verified advice"
            ]
        )

        custom_comment = st.text_input("Reviewer Notes / Commentary:", placeholder="Additional notes regarding this response...")
        combined_comment = f"{', '.join(comment_preset)}. {custom_comment}".strip(" .")

        if st.button("Save Review to Feedback Database", type="primary", use_container_width=True):
            if not review_q.strip() or not review_r.strip():
                st.warning("Question and response cannot be empty.")
            else:
                record_id = save_feedback(
                    question=review_q,
                    response=review_r,
                    risk_probability=review_risk,
                    model_decision="HUMAN_REVIEW",
                    human_decision=decision_code,
                    corrected_response=corrected_text if decision_code == "EDIT" else "",
                    reviewer_comment=combined_comment,
                    category=review_category
                )
                st.success(f"✅ Feedback successfully recorded in database (Record #{record_id})!")
                st.balloons()

# ---------------------------------------------------------
# TAB 3: ADAPTIVE RETRAINING & MODEL MANAGEMENT
# ---------------------------------------------------------
with tab3:
    st.header("Adaptive Learning & Model Retraining Pipeline")
    st.write("Incorporate human reviewer decisions to iteratively retrain, benchmark, and deploy improved risk classifiers.")

    df_feedback = get_feedback_df()
    
    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.subheader("Retraining Controls")
        st.write(f"**Baseline Samples:** 180 records")
        st.write(f"**Human Feedback Samples Available:** `{len(df_feedback)}` records")
        
        if len(df_feedback) < 3:
            st.info("💡 Tip: Use the 'Seed Sample Feedback' button in the sidebar or submit reviews in Tab 2 to provide training data.")

        if st.button("⚡ Trigger Adaptive Retraining Pipeline", type="primary", use_container_width=True):
            with st.spinner("Retraining candidate Random Forest classifier on combined dataset..."):
                try:
                    retrain_result = train_candidate_model()
                    st.session_state["retrain_result"] = retrain_result
                    st.success("Retraining & evaluation complete!")
                except Exception as ex:
                    st.error(f"Retraining failed: {ex}")

    with col_t2:
        st.subheader("Model Deployment & Versioning")
        
        if "retrain_result" in st.session_state:
            res = st.session_state["retrain_result"]
            cand_m = res["candidate_metrics"]
            prod_m = res["production_metrics"] or cand_m

            st.write("### Model Performance Comparison")
            comp_data = {
                "Metric": ["Accuracy", "ROC-AUC", "Recall (High Risk)", "Precision", "F1-Score", "5-Fold CV Accuracy"],
                "Production Model": [
                    f"{prod_m.get('accuracy', 0)*100:.2f}%",
                    f"{prod_m.get('roc_auc', 0):.4f}",
                    f"{prod_m.get('recall', 0)*100:.2f}%",
                    f"{prod_m.get('precision', 0)*100:.2f}%",
                    f"{prod_m.get('f1', 0):.4f}",
                    f"{prod_m.get('cv_mean_accuracy', prod_m.get('accuracy', 0))*100:.2f}%"
                ],
                "Candidate Model (Retrained)": [
                    f"{cand_m['accuracy']*100:.2f}%",
                    f"{cand_m['roc_auc']:.4f}",
                    f"{cand_m['recall']*100:.2f}%",
                    f"{cand_m['precision']*100:.2f}%",
                    f"{cand_m['f1']:.4f}",
                    f"{cand_m['cv_mean_accuracy']*100:.2f}%"
                ]
            }
            st.table(pd.DataFrame(comp_data))

            if res["meets_deployment_threshold"]:
                st.success("Candidate model meets quality and safety deployment thresholds.")
            else:
                st.warning("⚠️ Candidate model does not meet minimum safety thresholds.")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Deploy Candidate to Production", use_container_width=True):
                    dep = deploy_candidate_model()
                    st.success(f"Model deployed to production at {dep['timestamp']}!")
            with col_btn2:
                if st.button("⏮Rollback to Previous Model", use_container_width=True):
                    try:
                        rb = rollback_model()
                        st.info(f"Rolled back to previous backup at {rb['timestamp']}.")
                    except Exception as e:
                        st.warning(str(e))
        else:
            st.info("Click 'Trigger Adaptive Retraining Pipeline' to train and evaluate a new model.")

# ---------------------------------------------------------
# TAB 4: MONITORING & DATA DRIFT DASHBOARD
# ---------------------------------------------------------
with tab4:
    st.header("System Health, Drift Detection & Operational Monitoring")
    st.write("Track operational verification metrics and detect statistical data drift across incoming queries.")

    detector = DriftDetector()
    op_metrics = detector.get_operational_metrics()

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Human Reviews", op_metrics["total_reviewed_queries"])
    k2.metric("Approval Rate", f"{op_metrics['human_agreement_rate']*100:.1f}%")
    k3.metric("Rejection Rate", f"{op_metrics['rejection_rate']*100:.1f}%")
    k4.metric("Average Risk Score", f"{op_metrics['avg_risk_probability']:.2f}")

    st.divider()

    st.subheader("📈 Statistical Data Drift Analysis (Kolmogorov-Smirnov Test)")
    st.caption("Compares baseline training distributions against real-world reviewed queries. (Alpha = 0.05)")

    drift_report = detector.detect_feature_drift()
    
    if drift_report["status"] == "SUCCESS":
        if drift_report["drift_detected"]:
            st.warning(f"⚠️ Data Drift Detected in {drift_report['drift_features_count']} feature(s)! Model retraining recommended.")
        else:
            st.success("No statistically significant feature drift detected.")

        drift_rows = []
        for feat, d in drift_report["drift_results"].items():
            drift_rows.append({
                "Feature": feat,
                "Baseline Mean": d["baseline_mean"],
                "Runtime Mean": d["current_mean"],
                "% Shift": f"{d['mean_pct_change']:+.1f}%",
                "KS-Statistic": d["ks_statistic"],
                "p-value": d["p_value"],
                "Drift Detected?": "⚠️ YES" if d["has_drift"] else "✅ NO"
            })
        st.table(pd.DataFrame(drift_rows))
    else:
        st.info(drift_report["message"])

    st.divider()

    st.subheader("Review Feedback History & Audit Log")
    df_all_fb = get_feedback_df()
    
    if not df_all_fb.empty:
        filter_decision = st.selectbox("Filter by Human Decision:", ["ALL", "APPROVE", "REJECT", "EDIT"])
        if filter_decision != "ALL":
            filtered_df = df_all_fb[df_all_fb["human_decision"] == filter_decision]
        else:
            filtered_df = df_all_fb

        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Feedback Dataset (CSV)",
            data=csv_data,
            file_name="hitl_feedback_audit.csv",
            mime="text/csv"
        )
    else:
        st.write("No feedback records recorded yet.")

# ---------------------------------------------------------
# TAB 5: MODEL COMPARISON BENCHMARK
# ---------------------------------------------------------
with tab5:
    st.header("Multi-Model Benchmark Comparison")
    st.write("Compares 5 machine learning classifiers on the verification dataset using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.")

    if st.button("Run Multi-Model Benchmark", use_container_width=True):
        with st.spinner("Training and evaluating classifiers..."):
            comp_res = run_model_comparison()
            st.session_state["benchmark_res"] = comp_res

    if "benchmark_res" in st.session_state:
        b_res = st.session_state["benchmark_res"]
        st.success(f"Best performing classifier based on ROC-AUC: **{b_res['best_model']}**")
        st.dataframe(b_res["comparison_df"], use_container_width=True)
        st.bar_chart(b_res["comparison_df"].set_index("Model")[["Accuracy", "ROC-AUC", "Recall", "F1-Score"]])