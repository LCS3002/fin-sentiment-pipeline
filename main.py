"""Pipeline entry point: fetch articles, score sentiment, persist, report.

(Kept ASCII-only: argparse echoes this line to the console, and a Windows cp1252
terminal raises UnicodeEncodeError on characters like an arrow glyph.)

    python main.py AAPL
    python main.py AAPL MSFT --start 2024-01-01 --end 2024-03-31
    python main.py AAPL --earnings --summarize

Results are written to SQLite (`data/sentiment.db`) keyed on article url, so re-running
over an overlapping range updates rows rather than duplicating them.
"""

import argparse
import logging

import pandas as pd
from dotenv import load_dotenv

from config import COMMON_NAMES, SUMMARIZE, WINDOW_DAYS
from fetcher import EarningsFetcher, NewsFetcher
from processors import SentimentScorer
from processors.BART_summarizer import summarize_text
from storage import SentimentStore
from utils import setup_logging

logger = logging.getLogger(__name__)


def to_dataframe(results, ticker: str, summarize: bool = False) -> pd.DataFrame:
    """Raw Valyu results → a frame of [ticker, url, title, published_at, clean_text].

    `published_at` is parsed, and anything unparseable becomes NaT. The previous version
    stored the literal string "No Information" in this column, which silently poisoned
    any later `pd.to_datetime` and made missing dates indistinguishable from real ones.
    """
    company_name = COMMON_NAMES.get(ticker.upper(), ticker)
    rows = []

    for result in results:
        content = getattr(result, "content", "") or ""
        # content may arrive as a list of snippet dicts rather than flat text
        if isinstance(content, list):
            content = " ".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )

        title = getattr(result, "title", "") or ""
        full_text = f"{title}. {content}"

        # Keep only sentences naming the subject, so unrelated market commentary in the
        # same article does not get scored as if it were about this company.
        sentences = full_text.split(". ")
        relevant = [
            s
            for s in sentences
            if ticker.lower() in s.lower() or company_name.lower() in s.lower()
        ]
        clean_text = ". ".join(relevant) if relevant else title

        if summarize:
            clean_text = summarize_text(clean_text)

        rows.append(
            {
                "ticker": ticker.upper(),
                "url": getattr(result, "url", None),
                "title": title,
                "source": getattr(result, "source", None),
                "published_at": getattr(result, "publication_date", None),
                "clean_text": clean_text,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["ticker", "url", "title", "source", "published_at", "clean_text"],
    )
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        missing = int(df["published_at"].isna().sum())
        if missing:
            logger.warning(
                "%d of %d article(s) have no usable publication date - kept, but they "
                "cannot be aligned to a trading day",
                missing,
                len(df),
            )

    return df


def run(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    include_earnings: bool = False,
    summarize: bool = SUMMARIZE,
    db_path: str | None = None,
) -> pd.DataFrame:
    store = SentimentStore(db_path) if db_path else SentimentStore()
    frames = []

    for ticker in tickers:
        results = NewsFetcher(
            [ticker], start_date=start, end_date=end, window_days=WINDOW_DAYS
        ).fetch()

        if include_earnings:
            # Previously fetched and then thrown away without being used at all
            results += EarningsFetcher(
                [ticker], start_date=start, end_date=end
            ).fetch()

        df = to_dataframe(results, ticker, summarize=summarize)
        if df.empty:
            logger.warning("No articles for %s", ticker)
            continue
        frames.append(SentimentScorer(df))

    if not frames:
        logger.warning("Nothing scored")
        return pd.DataFrame()

    scored = pd.concat(frames, ignore_index=True)
    store.upsert(scored.to_dict("records"))

    daily = store.daily_sentiment()
    logger.info("Database now holds %d article(s)", store.count())

    print("\nDaily mean sentiment (score = p_positive - p_negative):")
    print(daily.to_string(index=False) if not daily.empty else "  (no dated articles)")

    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="+", help="one or more ticker symbols")
    parser.add_argument("--start", help="start date, YYYY-MM-DD (enables backfill)")
    parser.add_argument("--end", help="end date, YYYY-MM-DD")
    parser.add_argument(
        "--earnings", action="store_true", help="also fetch earnings reports"
    )
    parser.add_argument(
        "--summarize", action="store_true", help="summarise articles with BART first"
    )
    parser.add_argument("--db", help="SQLite path (default: data/sentiment.db)")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()

    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be given together")

    run(
        [t.upper() for t in args.tickers],
        start=args.start,
        end=args.end,
        include_earnings=args.earnings,
        summarize=args.summarize,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
