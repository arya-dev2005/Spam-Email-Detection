from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from core.config import BATCH_SIZE, MAX_LEN, THRESHOLD
from core.model_loader import load_artifacts
from core.preprocessor import clean_text


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    raw_prob: float
    ham_probability: float
    spam_probability: float
    cleaned_text: str
    tokens: list[str]
    token_scores: list[dict[str, float | str]]


def _prepare_sequences(texts: Iterable[str], tokenizer) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(list(texts))
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")


def _batch_predict_probabilities(texts: list[str], tokenizer, model) -> np.ndarray:
    if not texts:
        return np.array([], dtype=float)
    sequences = _prepare_sequences(texts, tokenizer)
    probabilities = model.predict(sequences, batch_size=min(BATCH_SIZE, len(texts)), verbose=0)
    return np.asarray(probabilities).reshape(-1)


def _token_scores(cleaned_text: str, tokenizer, model) -> list[dict[str, float | str]]:
    tokens = cleaned_text.split()
    if len(tokens) < 2:
        return []

    base_prob = float(_batch_predict_probabilities([cleaned_text], tokenizer, model)[0])
    variants = []
    variant_tokens = []
    for index, token in enumerate(tokens):
        perturbed = tokens[:index] + tokens[index + 1 :]
        if not perturbed:
            continue
        variants.append(" ".join(perturbed))
        variant_tokens.append(token)

    perturbed_probs = _batch_predict_probabilities(variants, tokenizer, model)
    scored_tokens = []
    for token, perturbed_prob in zip(variant_tokens, perturbed_probs, strict=False):
        score = abs(base_prob - float(perturbed_prob))
        scored_tokens.append({"token": token, "score": score})

    aggregated: dict[str, float] = {}
    for item in scored_tokens:
        aggregated[item["token"]] = max(aggregated.get(item["token"], 0.0), float(item["score"]))

    ranked = sorted(aggregated.items(), key=lambda pair: pair[1], reverse=True)
    return [{"token": token, "score": float(score)} for token, score in ranked[:12]]


def predict_text(text: str) -> PredictionResult:
    artifacts = load_artifacts()
    cleaned = clean_text(text)
    probability = float(_batch_predict_probabilities([cleaned], artifacts.tokenizer, artifacts.model)[0])
    label = "SPAM" if probability >= THRESHOLD else "HAM"
    confidence = probability if label == "SPAM" else 1.0 - probability
    scores = _token_scores(cleaned, artifacts.tokenizer, artifacts.model)
    return PredictionResult(
        label=label,
        confidence=float(confidence),
        raw_prob=float(probability),
        ham_probability=float(1.0 - probability),
        spam_probability=float(probability),
        cleaned_text=cleaned,
        tokens=cleaned.split(),
        token_scores=scores,
    )


def predict_dataframe(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    artifacts = load_artifacts()
    cleaned_texts = frame[text_column].astype(str).map(clean_text).tolist()
    probabilities = _batch_predict_probabilities(cleaned_texts, artifacts.tokenizer, artifacts.model)
    labels = np.where(probabilities >= THRESHOLD, "SPAM", "HAM")
    confidences = np.where(labels == "SPAM", probabilities, 1.0 - probabilities)
    output = frame.copy()
    output["cleaned_text"] = cleaned_texts
    output["spam_probability"] = probabilities
    output["ham_probability"] = 1.0 - probabilities
    output["prediction"] = labels
    output["confidence"] = confidences
    return output
