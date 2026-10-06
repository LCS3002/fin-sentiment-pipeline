"""Earnings-report fetching from Valyu's proprietary US earnings dataset.

Same windowing behaviour as `NewsFetcher` — see that module for why a date range has to
be requested in chunks rather than as one wide call.
"""

import logging
from typing import Optional

from config import END_DATE, START_DATE

from .valyu_client_class import ValyuClient
from .valyu_fetch_news import _parse, _windows

logger = logging.getLogger(__name__)

EARNINGS_SOURCE = "valyu/valyu-earnings-US"


class EarningsFetcher:
    def __init__(
        self,
        tickers: list[str],
        start_date: Optional[str] = START_DATE,
        end_date: Optional[str] = END_DATE,
        max_results: int = 10,
        window_days: int = 30,
    ) -> None:
        self.client = ValyuClient()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.max_results = max_results
        self.window_days = window_days

    def fetch(self) -> list:
        start, end = _parse(self.start_date), _parse(self.end_date)
        seen: set[str] = set()
        all_results = []

        for ticker in self.tickers:
            if start and end:
                windows = list(_windows(start, end, self.window_days))
                query = f"earnings reports about {ticker}"
            else:
                windows = [(None, None)]
                query = f"recent earnings reports about {ticker}"

            for window_start, window_end in windows:
                results = self.client.search(
                    query,
                    included_sources=[EARNINGS_SOURCE],
                    max_results=self.max_results,
                    start_date=window_start,
                    end_date=window_end,
                    search_type="all",  # proprietary dataset, not web news
                )
                for result in results:
                    url = getattr(result, "url", None)
                    if url and url in seen:
                        continue
                    if url:
                        seen.add(url)
                    all_results.append(result)

        logger.info(
            "Collected %d unique earnings record(s) for %s",
            len(all_results), ", ".join(self.tickers),
        )
        if not all_results:
            logger.warning("No earnings results found")

        return all_results
