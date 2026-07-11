from __future__ import annotations

import sys
from pathlib import Path
import uuid

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.database import store
from app.components.predictor import predict_text
from app.components.visualizations import apply_global_styles, confidence_gauge, probability_bars, result_card_html, token_scores_chart
from core.model_loader import load_artifacts

st.set_page_config(page_title="Detector | SpamShield AI", page_icon="🔍", layout="wide")
apply_global_styles()
load_artifacts()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

st.markdown("<div class='hero-card'><div class='hero-title' style='font-size:2.2rem;'>Real-Time Spam Detector</div><div class='hero-subtitle'>Paste an email or SMS message and get an instant label, calibrated confidence, probability breakdown, and token-level cues.</div></div>", unsafe_allow_html=True)

sample_ham = "Hey, are we still meeting for lunch at 1pm?"
sample_spam = "Congratulations! You have won a free prize. Click here to claim now."

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.text_area("Message text", value=sample_ham, height=220, placeholder="Paste a message to classify", key="message_input")
    button_row = st.columns(3)
    if button_row[0].button("Load ham sample", use_container_width=True):
        st.session_state.message_input = sample_ham
    if button_row[1].button("Load spam sample", use_container_width=True):
        st.session_state.message_input = sample_spam
    if button_row[2].button("Clear", use_container_width=True):
        st.session_state.message_input = ""

    message = st.session_state.get("message_input", "")
    run_prediction = st.button("Analyze message", type="primary", use_container_width=True)

with right:
    st.markdown("<div class='surface-card'><h3 style='margin-top:0;'>Prediction output</h3></div>", unsafe_allow_html=True)

if run_prediction and message.strip():
    result = predict_text(message)
    store.record_prediction(message, result.label, result.confidence, result.raw_prob, st.session_state.session_id)

    st.markdown(result_card_html(result.label, result.confidence, result.raw_prob), unsafe_allow_html=True)
    gauge_col, bar_col = st.columns([0.9, 1.1], gap="large")
    with gauge_col:
        st.plotly_chart(confidence_gauge(result.confidence), use_container_width=True)
    with bar_col:
        st.plotly_chart(probability_bars(result.ham_probability, result.spam_probability), use_container_width=True)

    st.subheader("Decision cues")
    st.plotly_chart(token_scores_chart(result.token_scores), use_container_width=True)

    st.caption("The token cue chart uses a leave-one-out sensitivity pass over the current message. It is a model-aware heuristic, not a native attention map.")
else:
    st.info("Enter a message and select Analyze message to run inference.")
