from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from core.config import DEFAULT_DB_PATH


class PredictionStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    raw_prob REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    total_rows INTEGER NOT NULL,
                    spam_count INTEGER NOT NULL,
                    ham_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL
                )
                """
            )

    def record_prediction(
        self,
        input_text: str,
        prediction: str,
        confidence: float,
        raw_prob: float,
        session_id: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO predictions (input_text, prediction, confidence, raw_prob, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (input_text, prediction, confidence, raw_prob, timestamp, session_id),
                )

    def record_batch_job(
        self,
        filename: str,
        total_rows: int,
        spam_count: int,
        ham_count: int,
        session_id: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO batch_jobs (filename, total_rows, spam_count, ham_count, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (filename, total_rows, spam_count, ham_count, timestamp, session_id),
                )

    def fetch_recent_predictions(self, limit: int = 200) -> pd.DataFrame:
        with self.connection() as connection:
            return pd.read_sql_query(
                """
                SELECT *
                FROM predictions
                ORDER BY datetime(timestamp) DESC, id DESC
                LIMIT ?
                """,
                connection,
                params=(limit,),
            )

    def fetch_summary(self) -> dict[str, float | int]:
        with self.connection() as connection:
            stats = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_predictions,
                    COALESCE(AVG(confidence), 0) AS avg_confidence,
                    COALESCE(SUM(CASE WHEN prediction = 'SPAM' THEN 1 ELSE 0 END), 0) AS spam_count,
                    COALESCE(SUM(CASE WHEN prediction = 'HAM' THEN 1 ELSE 0 END), 0) AS ham_count
                FROM predictions
                """
            ).fetchone()
        total_predictions = int(stats["total_predictions"] or 0)
        spam_count = int(stats["spam_count"] or 0)
        ham_count = int(stats["ham_count"] or 0)
        spam_rate = spam_count / total_predictions if total_predictions else 0.0
        return {
            "total_predictions": total_predictions,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "avg_confidence": float(stats["avg_confidence"] or 0.0),
            "spam_rate": spam_rate,
        }

    def fetch_daily_trend(self) -> pd.DataFrame:
        with self.connection() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    date(timestamp) AS day,
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN prediction = 'SPAM' THEN 1 ELSE 0 END) AS spam_count,
                    SUM(CASE WHEN prediction = 'HAM' THEN 1 ELSE 0 END) AS ham_count,
                    AVG(confidence) AS avg_confidence
                FROM predictions
                GROUP BY date(timestamp)
                ORDER BY day
                """,
                connection,
            )

    def fetch_batch_jobs(self) -> pd.DataFrame:
        with self.connection() as connection:
            return pd.read_sql_query(
                "SELECT * FROM batch_jobs ORDER BY datetime(timestamp) DESC, id DESC",
                connection,
            )


store = PredictionStore()
