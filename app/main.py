from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.database import store
from app.components.visualizations import apply_global_styles, dataset_badge
from core.model_loader import load_artifacts

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
apply_global_styles()

artifacts = load_artifacts()
summary = store.fetch_summary()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">SpamShield AI</div>
        <div class="hero-subtitle">A production-grade spam detection console powered by a Bidirectional LSTM, a reconstructed Keras tokenizer, and persistent SQLite history.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predictions", f"{summary['total_predictions']}")
col2.metric("Spam Rate", f"{summary['spam_rate'] * 100:.1f}%")
col3.metric("Average Confidence", f"{summary['avg_confidence'] * 100:.1f}%")
col4.metric("Training Accuracy", "98.92%")

st.write("")
left, right = st.columns([1.25, 0.95], gap="large")
with left:
    st.markdown(
        """
        <div class="surface-card">
            <h3 style="margin-top:0;">What this app includes</h3>
            <p style="color: rgba(240,246,252,0.80);">Real-time single-message detection, batch CSV analysis, historical analytics, and a model card that documents the BiLSTM architecture used to train the saved model.</p>
            <div>
                {ham}
                {spam}
                {db}
                {token}
            </div>
        </div>
        """.format(
            ham=dataset_badge("Ham / Spam classification"),
            spam=dataset_badge("Confidence scores"),
            db=dataset_badge("SQLite persistence"),
            token=dataset_badge("Tokenizer reconstruction"),
        ),
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="surface-card">
            <h3 style="margin-top:0;">Quick navigation</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_🔍_Detector.py", label="Open Detector", icon="🔍")
    st.page_link("pages/2_📊_Analytics.py", label="Open Analytics", icon="📊")
    st.page_link("pages/3_📂_Batch_Analysis.py", label="Open Batch Analysis", icon="📂")
    st.page_link("pages/4_ℹ️_About.py", label="Open About", icon="ℹ️")

st.write("")
with st.expander("Loaded model details", expanded=False):
    st.write(f"Model file: {artifacts.model_path}")
    st.write(f"Tokenizer cache: {artifacts.tokenizer_path}")
    st.code(artifacts.summary_text, language="text")
