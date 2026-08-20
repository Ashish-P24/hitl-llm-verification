import os
import sys

# Add the project root to Python's import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.llm_service import generate_response


st.set_page_config(
    page_title="HITL LLM Verification",
    layout="wide"
)

st.title("Human-in-the-Loop Adaptive Verification Framework")

st.write(
    "An adaptive framework that evaluates LLM responses "
    "and routes high-risk responses for human verification."
)

st.divider()

st.subheader("Ask the LLM")

question = st.text_area(
    "Enter your question",
    placeholder="Example: Who was the first person to walk on the Moon?"
)

if st.button("Generate Response"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Generating response..."):

            try:
                answer = generate_response(question)

                st.subheader("LLM Response")
                st.write(answer)

            except Exception as error:
                st.error(f"Error: {error}")