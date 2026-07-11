from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

from core.config import (
    DATASET_CANDIDATES,
    MODEL_CANDIDATES,
    OOV_TOKEN,
    RANDOM_STATE,
    TOKENIZER_PATH,
    VOCAB_SIZE,
)
from core.preprocessor import clean_texts, standardize_dataset


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    tokenizer: Tokenizer
    training_frame: pd.DataFrame
    model_path: Path
    tokenizer_path: Path
    summary_text: str


def _find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {', '.join(str(path) for path in paths)}")


def _load_training_frame() -> pd.DataFrame:
    dataset_path = _find_first_existing(DATASET_CANDIDATES)
    raw_frame = pd.read_csv(dataset_path, encoding="latin-1")
    return standardize_dataset(raw_frame)


def _build_tokenizer(training_texts: list[str]) -> Tokenizer:
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(training_texts)
    return tokenizer


def _load_or_create_tokenizer(training_frame: pd.DataFrame) -> Tokenizer:
    if TOKENIZER_PATH.exists():
        with TOKENIZER_PATH.open("rb") as handle:
            return pickle.load(handle)

    X_train, _, _, _ = train_test_split(
        training_frame["clean_message"],
        training_frame["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    tokenizer = _build_tokenizer(clean_texts(X_train))
    with TOKENIZER_PATH.open("wb") as handle:
        pickle.dump(tokenizer, handle)
    return tokenizer


def _capture_model_summary(model: object) -> str:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
    return buffer.getvalue().strip()


@st.cache_resource(show_spinner=False)
def load_artifacts() -> ModelArtifacts:
    training_frame = _load_training_frame()
    model_path = _find_first_existing(MODEL_CANDIDATES)
    model = load_model(model_path)
    tokenizer = _load_or_create_tokenizer(training_frame)
    summary_text = _capture_model_summary(model)
    return ModelArtifacts(
        model=model,
        tokenizer=tokenizer,
        training_frame=training_frame,
        model_path=model_path,
        tokenizer_path=TOKENIZER_PATH,
        summary_text=summary_text,
    )
