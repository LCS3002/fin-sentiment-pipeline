"""SQLite persistence for scored articles.

Uses the stdlib `sqlite3` — no new dependency, and a single file is the right weight for
this. The article URL is the primary key, so re-running over an overlapping date range
updates rows instead of duplicating them: Valyu's search is relevance-ranked rather than
chronological, so overlapping windows returning the same article is normal, not an error.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/sentiment.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url           TEXT PRIMARY KEY,
    ticker        TEXT NOT NULL,
    published_at  TEXT,              -- ISO-8601 date, or NULL when the source omits it
    title         TEXT,
    clean_text    TEXT,
    sentiment     INTEGER,           -- -1 negative / 0 neutral / 1 positive
    p_negative    REAL,
    p_neutral     REAL,
    p_positive    REAL,
    score         REAL,              -- p_positive - p_negative, the continuous signal
    confidence    REAL,              -- probability of the winning label
    source        TEXT,
    fetched_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_ticker_date ON articles (ticker, published_at);
"""

_COLUMNS = (
    "url", "ticker", "published_at", "title", "clean_text", "sentiment",
    "p_negative", "p_neutral", "p_positive", "score", "confidence", "source",
    "fetched_at",
)


class SentimentStore:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert(self, rows: Iterable[dict]) -> int:
        """Insert or replace articles keyed on `url`. Returns the number written."""
        payload = []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for row in rows:
            url = row.get("url")
            if not url:
                logger.debug("Skipping a row with no url - nothing to key it on")
                continue
            record = {col: row.get(col) for col in _COLUMNS}
            record["url"] = url
            record["fetched_at"] = now
            # pandas NaT/NaN are not sqlite-friendly; store a real NULL instead
            published = record.get("published_at")
            if published is None or pd.isna(published):
                record["published_at"] = None
            elif isinstance(published, (pd.Timestamp, datetime)):
                record["published_at"] = published.date().isoformat()
            payload.append(tuple(record[col] for col in _COLUMNS))

        if not payload:
            return 0

        placeholders = ", ".join("?" * len(_COLUMNS))
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO articles ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                payload,
            )

        logger.info("Wrote %d article(s) to %s", len(payload), self.path)
        return len(payload)

    def load(
        self, ticker: Optional[str] = None, start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Scored articles as a DataFrame, optionally filtered by ticker and date."""
        clauses, params = [], []
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker.upper())
        if start:
            clauses.append("published_at >= ?")
            params.append(start)
        if end:
            clauses.append("published_at <= ?")
            params.append(end)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM articles{where} ORDER BY published_at, url"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        return df

    def daily_sentiment(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Mean continuous score per ticker-day, with the article count behind it.

        Articles with no publication date are dropped rather than bucketed into an
        arbitrary day — an unknown date cannot be aligned to a trading session.
        """
        df = self.load(ticker=ticker).dropna(subset=["published_at"])
        if df.empty:
            return pd.DataFrame(columns=["ticker", "date", "score", "articles"])

        out = (
            df.assign(date=df["published_at"].dt.date)
            .groupby(["ticker", "date"], as_index=False)
            .agg(score=("score", "mean"), articles=("url", "count"))
        )
        return out

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
