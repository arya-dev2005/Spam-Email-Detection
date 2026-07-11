from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

_TEXT_CLEAN_RE = re.compile(r"[^a-z0-9']+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    normalized = str(text).lower().replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = _TEXT_CLEAN_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def standardize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    columns = {column.lower(): column for column in renamed.columns}

    if {"category", "message"}.issubset(columns):
        renamed = renamed[[columns["category"], columns["message"]]].copy()
        renamed.columns = ["label", "message"]
    elif {"v1", "v2"}.issubset(columns):
        renamed = renamed[[columns["v1"], columns["v2"]]].copy()
        renamed.columns = ["label", "message"]
    elif {"label", "message"}.issubset(columns):
        renamed = renamed[[columns["label"], columns["message"]]].copy()
    else:
        raise ValueError("Unsupported spam dataset schema. Expected Category/Message, v1/v2, or label/message columns.")

    renamed = renamed.dropna(subset=["label", "message"]).copy()
    renamed["label"] = renamed["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    renamed = renamed.dropna(subset=["label"]).copy()
    renamed["label"] = renamed["label"].astype(int)
    renamed["message"] = renamed["message"].astype(str)
    renamed["clean_message"] = renamed["message"].map(clean_text)
    renamed = renamed[renamed["clean_message"].str.len() > 0].copy()
    renamed.reset_index(drop=True, inplace=True)
    return renamed


def clean_texts(texts: Iterable[str]) -> list[str]:
    return [clean_text(text) for text in texts]
