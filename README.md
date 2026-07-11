# SpamShield AI

SpamShield AI is a Streamlit interface for the saved BiLSTM spam classifier in this workspace. It provides single-message detection, batch scoring, analytics, and an about page with model metadata.

## Run

1. Install dependencies from `requirements.txt`.
2. Launch the app with:

```bash
streamlit run app/main.py
```

## Notes

- The app reconstructs the Keras tokenizer from `spam.csv` on first startup and caches it to `models/tokenizer.pkl`.
- The app accepts either the Kaggle-style `Category/Message` CSV layout or the earlier `v1/v2` schema.
- Existing training artifacts are read from the workspace root if they are not already copied into `models/` or `results/`.

## Streamlit Community Cloud

This repository is configured for the free Streamlit Community Cloud tier.

Deploy with:

1. Repository root: this project folder.
2. App file: `app/main.py`.
3. Runtime: `runtime.txt` pins Python 3.11 for better compatibility on Community Cloud.
4. Streamlit config: `.streamlit/config.toml` enables headless Cloud-friendly defaults and keeps usage stats disabled.
