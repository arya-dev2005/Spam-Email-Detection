from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.visualizations import apply_global_styles, dataset_badge
from core.model_loader import load_artifacts

st.set_page_config(page_title="About | SpamShield AI", page_icon="ℹ️", layout="wide")
apply_global_styles()
artifacts = load_artifacts()
frame = artifacts.training_frame

st.markdown("<div class='hero-card'><div class='hero-title' style='font-size:2.2rem;'>About SpamShield AI</div><div class='hero-subtitle'>A compact model card for the BiLSTM spam classifier, its training data, and the runtime stack used by the app.</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown(
        f"""
        <div class="surface-card">
            <h3 style="margin-top:0;">Model card</h3>
            <p><strong>Architecture:</strong> Embedding(10000, 128) → BiLSTM(64) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)</p>
            <p><strong>Tokenizer:</strong> Keras Tokenizer refit from the original training corpus at startup and cached to disk after the first load.</p>
            <p><strong>Inference length:</strong> 100 tokens</p>
            <p><strong>Saved model:</strong> {artifacts.model_path.name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="surface-card">
            <h3 style="margin-top:0;">Training data</h3>
            <p><strong>Source:</strong> Kaggle spam email dataset</p>
            <p><strong>Rows:</strong> {len(frame):,}</p>
            <p><strong>Ham rows:</strong> {int((frame['label'] == 0).sum()):,}</p>
            <p><strong>Spam rows:</strong> {int((frame['label'] == 1).sum()):,}</p>
            <div>{dataset_badge('Balanced evaluation split')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="surface-card">
            <h3 style="margin-top:0;">Tech stack</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for badge in ["Streamlit", "TensorFlow / Keras", "SQLite", "Plotly", "Pandas", "Scikit-learn"]:
        st.markdown(dataset_badge(badge), unsafe_allow_html=True)

st.subheader("Performance snapshot")
metrics = pd.DataFrame(
    {
        "Metric": ["Accuracy", "F1 (spam)", "Precision (spam)", "Recall (spam)"],
        "Value": ["98.92%", "High 0.98+", "High 0.98+", "High 0.98+"],
    }
)
st.dataframe(metrics, use_container_width=True, hide_index=True)

st.subheader("Repository layout")
st.code(
    """
app/
core/
models/
dataset/
notebook/
results/
data/
    """.strip(),
    language="text",
)
