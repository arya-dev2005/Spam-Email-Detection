from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import RESULT_IMAGE_PATHS


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29,185,84,0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(229,57,53,0.12), transparent 24%),
                linear-gradient(180deg, #0E1117 0%, #0B0F14 100%);
        }
        .hero-card, .surface-card {
            background: rgba(22, 27, 34, 0.88);
            border: 1px solid rgba(240, 246, 252, 0.08);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.32);
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.25rem;
            letter-spacing: -0.04em;
        }
        .hero-subtitle {
            color: rgba(240, 246, 252, 0.76);
            font-size: 1.02rem;
            max-width: 70ch;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            margin: 0.15rem 0.3rem 0.15rem 0;
            background: rgba(240, 246, 252, 0.08);
            color: #F0F6FC;
            border: 1px solid rgba(240, 246, 252, 0.09);
            font-size: 0.83rem;
        }
        .result-card {
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            border: 1px solid rgba(240, 246, 252, 0.12);
            animation: pulseIn 0.7s ease-out;
        }
        .result-spam {
            background: linear-gradient(135deg, rgba(229,57,53,0.28), rgba(229,57,53,0.10));
        }
        .result-ham {
            background: linear-gradient(135deg, rgba(29,185,84,0.26), rgba(29,185,84,0.10));
        }
        @keyframes pulseIn {
            from { transform: translateY(8px) scale(0.985); opacity: 0.2; }
            to { transform: translateY(0) scale(1); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def confidence_gauge(probability: float, title: str = "Confidence") -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1DB954" if probability < 0.5 else "#E53935"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(29,185,84,0.15)"},
                    {"range": [50, 100], "color": "rgba(229,57,53,0.15)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": probability * 100},
            },
        )
    )
    figure.update_layout(height=280, margin={"l": 20, "r": 20, "t": 40, "b": 20}, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#F0F6FC"})
    return figure


def probability_bars(ham_probability: float, spam_probability: float) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Bar(name="HAM", x=["Probability"], y=[ham_probability * 100], marker_color="#1DB954"),
            go.Bar(name="SPAM", x=["Probability"], y=[spam_probability * 100], marker_color="#E53935"),
        ]
    )
    figure.update_layout(barmode="group", height=320, yaxis_title="Probability (%)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F0F6FC"})
    return figure


def token_scores_chart(token_scores: list[dict[str, float | str]]) -> go.Figure:
    if not token_scores:
        figure = go.Figure()
        figure.add_annotation(text="No token-level signals available for this input.", showarrow=False, font={"color": "#F0F6FC"})
        figure.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return figure

    frame = pd.DataFrame(token_scores)
    frame = frame.sort_values("score", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=frame["score"],
            y=frame["token"],
            orientation="h",
            marker_color=frame["score"],
            marker_colorscale=[[0, "#1DB954"], [1, "#E53935"]],
        )
    )
    figure.update_layout(height=320, margin={"l": 10, "r": 10, "t": 15, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F0F6FC"})
    return figure


def trend_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        figure = go.Figure()
        figure.add_annotation(text="No prediction history yet.", showarrow=False, font={"color": "#F0F6FC"})
        figure.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)")
        return figure

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame["day"], y=frame["total_predictions"], name="Total", mode="lines+markers", line={"color": "#7C3AED", "width": 3}))
    figure.add_trace(go.Scatter(x=frame["day"], y=frame["spam_count"], name="Spam", mode="lines+markers", line={"color": "#E53935", "width": 3}))
    figure.add_trace(go.Scatter(x=frame["day"], y=frame["ham_count"], name="Ham", mode="lines+markers", line={"color": "#1DB954", "width": 3}))
    figure.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F0F6FC"}, legend={"orientation": "h"})
    return figure


def confusion_matrix_figure(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[[0, "#0E1117"], [0.5, "#1DB954"], [1, "#E53935"]],
            showscale=True,
        )
    )
    figure.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F0F6FC"})
    return figure


def dataset_badge(label: str) -> str:
    return f'<span class="pill">{label}</span>'


def result_card_html(label: str, confidence: float, raw_prob: float) -> str:
    tone = "result-spam" if label == "SPAM" else "result-ham"
    verb = "High risk message" if label == "SPAM" else "Likely safe message"
    return f"""
    <div class="result-card {tone}">
        <div style="font-size: 0.85rem; opacity: 0.85; letter-spacing: 0.12em; text-transform: uppercase;">Prediction Result</div>
        <div style="font-size: 2rem; font-weight: 800; margin-top: 0.25rem;">{label}</div>
        <div style="font-size: 1.05rem; margin-top: 0.25rem;">{verb}</div>
        <div style="margin-top: 0.55rem; font-size: 0.96rem; color: rgba(240,246,252,0.86);">Confidence: {confidence * 100:.1f}% | Spam probability: {raw_prob * 100:.1f}%</div>
    </div>
    """


def image_path(key: str) -> Path:
    return RESULT_IMAGE_PATHS[key]
