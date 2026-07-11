from __future__ import annotations

import io
import sys
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.database import store
from app.components.predictor import predict_dataframe
from app.components.visualizations import apply_global_styles, dataset_badge, probability_bars
from core.model_loader import load_artifacts

st.set_page_config(page_title="Batch Analysis | SpamShield AI", page_icon="📂", layout="wide")
apply_global_styles()
load_artifacts()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

st.markdown("<div class='hero-card'><div class='hero-title' style='font-size:2.2rem;'>Batch Analysis</div><div class='hero-subtitle'>Upload a CSV file, pick the text column, and classify every row with downloadable results and aggregate stats.</div></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to start bulk inference.")
    st.stop()

frame = pd.read_csv(uploaded_file)
st.write("Preview")
st.dataframe(frame.head(20), use_container_width=True)

candidate_columns = [column for column in frame.columns if frame[column].dtype == "object"] or list(frame.columns)
text_column = st.selectbox("Text column", candidate_columns)

if st.button("Run batch inference", type="primary", use_container_width=True):
    with st.spinner("Scoring rows..."):
        results = predict_dataframe(frame, text_column)
    spam_count = int((results["prediction"] == "SPAM").sum())
    ham_count = int((results["prediction"] == "HAM").sum())
    store.record_batch_job(uploaded_file.name, len(results), spam_count, ham_count, st.session_state.session_id)

    total = len(results)
    spam_rate = spam_count / total if total else 0.0
    confidence = float(results["confidence"].mean()) if total else 0.0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows processed", f"{total}")
    metric_cols[1].metric("Spam rows", f"{spam_count}")
    metric_cols[2].metric("Ham rows", f"{ham_count}")
    metric_cols[3].metric("Mean confidence", f"{confidence * 100:.1f}%")

    summary_cols = st.columns([1, 1], gap="large")
    with summary_cols[0]:
        st.plotly_chart(probability_bars(1.0 - spam_rate, spam_rate), use_container_width=True)
    with summary_cols[1]:
        st.write(pd.DataFrame({"label": ["HAM", "SPAM"], "count": [ham_count, spam_count]}))

    buffer = io.StringIO()
    results.to_csv(buffer, index=False)
    st.download_button(
        "Download predictions CSV",
        data=buffer.getvalue(),
        file_name="spamshield_batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Results preview")
    st.dataframe(results.head(50), use_container_width=True)
    st.caption(dataset_badge("Results include prediction, confidence, and probability columns"), unsafe_allow_html=True)
