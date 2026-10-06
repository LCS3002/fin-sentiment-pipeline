"""FinBERT sentiment scoring.

Returns the full probability distribution rather than just the winning label. The
previous version collapsed every article to {-1, 0, 1}, which throws away most of the
signal — a 0.51-confidence positive and a 0.99-confidence positive became the same
number. The continuous `score` (p_positive - p_negative) is what a factor can use.

A caveat worth knowing before trusting these labels: `ProsusAI/finbert` is itself
fine-tuned on the Financial PhraseBank, so evaluating it against that benchmark measures
nothing. Treat its output as a feature, not as ground truth.
"""

import logging
from functools import lru_cache

import pandas as pd

from config import SENTIMENT_MODEL

logger = logging.getLogger(__name__)

LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}
_PROB_COLUMNS = ["p_negative", "p_neutral", "p_positive"]
_OUTPUT_COLUMNS = _PROB_COLUMNS + ["sentiment", "confidence", "score"]


@lru_cache(maxsize=1)
def _model():
    """Loaded on first use — importing this module should not pull in model weights.

    `top_k=None` makes the pipeline return every class with its probability instead of
    only the argmax.
    """
    from transformers import pipeline  # imported here to keep module import cheap

    logger.info("Loading sentiment model %s", SENTIMENT_MODEL)
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=None)


def _distribution(scores: list[dict]) -> dict:
    """One article's class scores → a fixed-shape probability dict."""
    probs = {"p_negative": 0.0, "p_neutral": 0.0, "p_positive": 0.0}

    for entry in scores:
        label = str(entry.get("label", "")).lower()
        if label in LABEL_MAP:
            probs[f"p_{label}"] = float(entry.get("score", 0.0))
        else:
            # Never KeyError on a label this model was not expected to emit
            logger.warning("Unrecognised sentiment label %r - ignored", entry.get("label"))

    return probs


def SentimentScorer(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Score `clean_text` and append probability, label, confidence and score columns.

    The input frame is returned with the new columns added, so callers keep whatever
    identifying fields (url, title) they were carrying.
    """
    df = df.copy()

    if df.empty or "clean_text" not in df.columns:
        logger.warning("Nothing to score - returning empty columns")
        for col in _OUTPUT_COLUMNS:
            df[col] = pd.Series(dtype="float64")
        return df

    texts = df["clean_text"].fillna("").astype(str).tolist()
    results = _model()(texts, batch_size=batch_size)

    rows = []
    for scores in results:
        # with top_k=None the pipeline yields a list per input; tolerate a bare dict
        if isinstance(scores, dict):
            scores = [scores]
        probs = _distribution(scores)
        winner = max(LABEL_MAP, key=lambda label: probs[f"p_{label}"])
        rows.append(
            {
                **probs,
                "sentiment": LABEL_MAP[winner],
                "confidence": probs[f"p_{winner}"],
                "score": probs["p_positive"] - probs["p_negative"],
            }
        )

    scored = pd.DataFrame(rows, index=df.index)
    for col in _OUTPUT_COLUMNS:
        df[col] = scored[col]

    logger.info("Scored %d article(s)", len(df))
    return df
