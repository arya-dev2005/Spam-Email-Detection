# SpamShield AI

SpamShield AI is a Streamlit application for spam email and SMS detection powered by a TensorFlow/Keras Bidirectional LSTM model. The app provides a polished end-user interface on top of the research notebook, with real-time detection, batch inference, analytics, and persistent prediction history.

The model was trained on the spam email dataset and achieved approximately 98.92% accuracy in the notebook workflow.

## Features

- Real-time spam detection for a single message
- Batch CSV analysis with downloadable results
- Confidence scoring and ham/spam probability breakdowns
- Token-level sensitivity cues for the current prediction
- Historical analytics backed by SQLite
- Model architecture summary and training artifact previews
- Streamlit Community Cloud deployment support

## Model Overview

- Text preprocessing: lowercase normalization and punctuation cleanup
- Tokenization: Keras Tokenizer refit from the training corpus at startup
- Sequence length: 100 tokens
- Architecture: Embedding(10000, 128) → BiLSTM(64) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Saved model: `best_spam_model.keras`

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Matplotlib
- Seaborn
- SQLite

## Project Structure

```text
Spam_Email_Detection/
├── app/
│   ├── main.py
│   ├── pages/
│   └── components/
├── core/
├── dataset/
├── models/
├── notebook/
├── results/
├── data/
├── requirements.txt
├── runtime.txt
└── .streamlit/config.toml
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the app locally:

```bash
streamlit run app/main.py
```

3. Open the app in your browser and use the Detector, Analytics, Batch Analysis, and About pages from the sidebar.

## Notebook Reference

The research notebook is available in this repository at [notebook/spam_email_detection.ipynb](notebook/spam_email_detection.ipynb).

Kaggle notebook reference:

https://www.kaggle.com/code/aryadev25/spam-email-detection

## Deployment

This repository is configured for the free Streamlit Community Cloud tier.

- App entry point: `app/main.py`
- Python runtime: `runtime.txt` pins Python 3.11
- Streamlit settings: `.streamlit/config.toml` enables headless, Cloud-friendly defaults

## Notes

- The app reconstructs the Keras tokenizer from `spam.csv` on first startup and caches it to `models/tokenizer.pkl`.
- The app accepts either the Kaggle-style `Category/Message` schema or the older `v1/v2` schema.
- Training artifacts such as the confusion matrix and training curves are stored in `results/`.
- Prediction history is stored locally in `data/spamshield.db`.
