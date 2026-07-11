from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.database import store
from app.components.predictor import predict_dataframe
from app.components.visualizations import apply_global_styles, confusion_matrix_figure, image_path, trend_chart
from core.config import RANDOM_STATE
from core.model_loader import load_artifacts

st.set_page_config(page_title="Analytics | SpamShield AI", page_icon="📊", layout="wide")
apply_global_styles()
artifacts = load_artifacts()
summary = store.fetch_summary()
trend_frame = store.fetch_daily_trend()

st.markdown("<div class='hero-card'><div class='hero-title' style='font-size:2.2rem;'>Analytics Dashboard</div><div class='hero-subtitle'>Monitor prediction volume, confidence, historical behavior, and model quality on the original spam dataset.</div></div>", unsafe_allow_html=True)

metric_cols = st.columns(4)
metric_cols[0].metric("Total Predictions", f"{summary['total_predictions']}")
metric_cols[1].metric("Spam Predictions", f"{summary['spam_count']}")
metric_cols[2].metric("Ham Predictions", f"{summary['ham_count']}")
metric_cols[3].metric("Average Confidence", f"{summary['avg_confidence'] * 100:.1f}%")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Historical trend")
    st.plotly_chart(trend_chart(trend_frame), use_container_width=True)
with right:
    st.subheader("Training artifacts")
    st.image(str(image_path("training_curves")), use_container_width=True)
    st.caption("Original training curve snapshot preserved from the research notebook.")

st.subheader("Model quality on reconstructed tokenizer split")
reconstructed = artifacts.training_frame.copy()
train_frame, test_frame = train_test_split(reconstructed, test_size=0.2, random_state=RANDOM_STATE)
prediction_frame = predict_dataframe(test_frame, "message")
true_labels = test_frame["label"].to_numpy()
predicted_labels = prediction_frame["prediction"].map({"HAM": 0, "SPAM": 1}).to_numpy()
report = classification_report(true_labels, predicted_labels, target_names=["Ham", "Spam"], output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

report_frame = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
report_frame = report_frame[["class", "precision", "recall", "f1-score", "support"]]
st.dataframe(report_frame, use_container_width=True, hide_index=True)

matrix_col, image_col = st.columns([1, 1], gap="large")
with matrix_col:
    st.plotly_chart(confusion_matrix_figure(conf_matrix.tolist(), ["Ham", "Spam"]), use_container_width=True)
with image_col:
    st.image(str(image_path("confusion_matrix")), use_container_width=True)
    st.caption("Saved confusion matrix artifact from the training run.")

st.subheader("Model architecture")
st.code(artifacts.summary_text, language="text")
