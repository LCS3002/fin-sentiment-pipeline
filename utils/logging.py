"""Logging setup — one call site, so modules only ever do `logging.getLogger(__name__)`.

The pipeline previously reported progress with bare `print()` calls, which gave no
levels, no timestamps and no way to quieten the noisy parts. Transformers in particular
is chatty on import.
"""

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Level comes from LOG_LEVEL, defaulting to INFO."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Model downloads and tokenizer chatter drown out the pipeline's own output
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True
