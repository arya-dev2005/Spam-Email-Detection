from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
CORE_DIR = PROJECT_ROOT / "core"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
DEFAULT_DB_PATH = DATA_DIR / "spamshield.db"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
MODEL_CANDIDATES = [
    MODELS_DIR / "best_spam_model.keras",
    PROJECT_ROOT / "best_spam_model.keras",
]
DATASET_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "spam.csv",
    PROJECT_ROOT / "spam.csv",
    DATA_DIR / "spam.csv",
]
RESULT_IMAGE_PATHS = {
    "classification_report": PROJECT_ROOT / "classification_report.png",
    "confusion_matrix": PROJECT_ROOT / "confusion_matrix.png",
    "training_curves": PROJECT_ROOT / "training_curves.png",
}
VOCAB_SIZE = 10_000
MAX_LEN = 100
OOV_TOKEN = "<OOV>"
RANDOM_STATE = 42
THRESHOLD = 0.5
BATCH_SIZE = 128
