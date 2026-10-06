"""Thin wrapper around the Valyu SDK.

Two things worth knowing about the upstream API:

  · `start_date` / `end_date` are `YYYY-MM-DD` strings, so results carry **date-level**
    granularity at best. `publication_date` comes back as an unvalidated string and can
    be absent entirely — callers must treat a missing date as missing, never default it.

  · `search` is a *relevance-ranked* search, not a chronological archive. A wide date
    range returns the N most relevant hits in that window, not every article in it. To
    cover a period properly, request it in narrow windows and de-duplicate on `url`.
"""

import logging
from typing import Optional

from valyu import Valyu

logger = logging.getLogger(__name__)


class ValyuClient:
    def __init__(self) -> None:
        self.client = Valyu()
        self.total_cost_usd = 0.0

    def search(
        self,
        query: str,
        included_sources: Optional[list[str]] = None,
        max_results: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search_type: str = "news",
        relevance_threshold: float = 0.5,
    ) -> list:
        """Run one search. Returns results, or an empty list on failure.

        Failures are logged with their query and window — the previous version printed a
        bare message, which made an empty result indistinguishable from a quiet error
        during a long backfill.
        """
        try:
            response = self.client.search(
                query=query,
                included_sources=included_sources or [],
                max_num_results=max_results,
                start_date=start_date,
                end_date=end_date,
                search_type=search_type,
                relevance_threshold=relevance_threshold,
            )
        except Exception as e:
            logger.error(
                "Valyu search failed (query=%r window=%s..%s): %s",
                query, start_date, end_date, e,
            )
            return []

        if response is None:
            logger.error("Valyu returned no response (query=%r)", query)
            return []

        if getattr(response, "success", True) is False:
            logger.error(
                "Valyu reported failure (query=%r): %s", query,
                getattr(response, "error", "no error message"),
            )
            return []

        # Track spend — a multi-window backfill is many billed calls
        cost = getattr(response, "total_deduction_dollars", None)
        if cost:
            self.total_cost_usd += float(cost)

        results = getattr(response, "results", None) or []
        logger.debug(
            "Valyu returned %d result(s) (query=%r window=%s..%s)",
            len(results), query, start_date, end_date,
        )
        return results
