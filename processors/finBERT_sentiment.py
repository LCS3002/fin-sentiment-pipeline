import pandas as pd
from transformers import pipeline
from config import SENTIMENT_MODEL

sentiment_model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}


# Score each row's clean_text with FinBERT and return sentiment + confidence columns.
def SentimentScorer(df: pd.DataFrame) -> pd.DataFrame:
    results = sentiment_model(df["clean_text"].tolist(), batch_size=32)

    df["sentiment"] = [LABEL_MAP[r["label"].lower()] for r in results]
    df["confidence"] = [r["score"] for r in results]

    return df[["ticker", "timestamp", "sentiment", "confidence"]]
