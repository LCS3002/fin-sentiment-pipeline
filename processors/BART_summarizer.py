"""Optional BART summarisation.

Off by default (`config.SUMMARIZE`). It is one model call per article, which dominates
runtime on any sizeable fetch, and FinBERT truncates to 512 tokens anyway — so for most
articles it costs time and introduces a second model's bias without adding information.
Useful when articles are long and you want the lede rather than a truncation.

The model is loaded on first call, not at import: this module used to pull ~1.6GB of
weights into memory even when summarisation was never used.
"""

import logging
from functools import lru_cache

from config import SUMMARIZER_MODEL

logger = logging.getLogger(__name__)

_MIN_WORDS = 50


@lru_cache(maxsize=1)
def _summarizer():
    from transformers import pipeline  # local import keeps module import cheap

    logger.info("Loading summarisation model %s", SUMMARIZER_MODEL)
    return pipeline("summarization", model=SUMMARIZER_MODEL)


def summarize_text(text: str) -> str:
    """Summarise, or return the input unchanged if it is short or summarisation fails."""
    word_count = len(text.split())
    if word_count < _MIN_WORDS:
        return text

    try:
        # Cap max_length below input length to avoid HuggingFace length warnings
        max_len = min(60, int(word_count * 0.8))
        min_len = min(20, int(max_len * 0.5))

        summary = _summarizer()(
            text, max_length=max_len, min_length=min_len, do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception as e:
        logger.warning("Summarisation failed, using original text: %s", e)
        return text
