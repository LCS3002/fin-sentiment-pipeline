"""News fetching, optionally over a historical date range.

`config.START_DATE` / `END_DATE` existed but were never wired to anything, so the fetcher
only ever returned current news. They are now honoured, and because Valyu ranks by
relevance rather than date, a range is requested as a series of narrow windows and
de-duplicated on url — one wide request would just return the N most relevant hits for
the whole period.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Iterator, Optional

from config import DEFAULT_NEWS_SOURCES, END_DATE, START_DATE

from .valyu_client_class import ValyuClient

logger = logging.getLogger(__name__)


def _parse(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _windows(start: date, end: date, days: int) -> Iterator[tuple[str, str]]:
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        yield cursor.isoformat(), stop.isoformat()
        cursor = stop + timedelta(days=1)


class NewsFetcher:
    """Fetches news for the given tickers.

    With no dates, returns current news (the original behaviour). With dates, walks the
    range in `window_days` chunks.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: Optional[str] = START_DATE,
        end_date: Optional[str] = END_DATE,
        max_results: int = 10,
        window_days: int = 7,
        sources: Optional[list[str]] = None,
    ) -> None:
        self.client = ValyuClient()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.max_results = max_results
        self.window_days = window_days
        self.sources = sources if sources is not None else list(DEFAULT_NEWS_SOURCES)

    def fetch(self) -> list:
        start, end = _parse(self.start_date), _parse(self.end_date)
        seen: set[str] = set()
        all_results = []

        for ticker in self.tickers:
            if start and end:
                windows = list(_windows(start, end, self.window_days))
                query = f"news about {ticker}"
                logger.info(
                    "Fetching %s across %d window(s) of %d day(s): %s..%s",
                    ticker, len(windows), self.window_days, start, end,
                )
            else:
                windows = [(None, None)]
                query = f"latest news about {ticker}"

            for window_start, window_end in windows:
                results = self.client.search(
                    query,
                    included_sources=self.sources,
                    max_results=self.max_results,
                    start_date=window_start,
                    end_date=window_end,
                )
                for result in results:
                    url = getattr(result, "url", None)
                    if url and url in seen:
                        continue
                    if url:
                        seen.add(url)
                    all_results.append(result)

        logger.info(
            "Collected %d unique article(s) for %s (Valyu spend $%.4f)",
            len(all_results), ", ".join(self.tickers), self.client.total_cost_usd,
        )
        if not all_results:
            logger.warning("No news results found - check the date range and sources")

        return all_results
