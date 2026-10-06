"""Static configuration. Secrets live in `.env` and are read by the Valyu SDK itself."""

# Ticker -> company name lookup, used to keep only sentences that mention the subject
COMMON_NAMES = {
    "AAPL": "Apple",
    "META": "Meta",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
}

# Models
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
SENTIMENT_MODEL = "ProsusAI/finbert"

# Domains to restrict news search to. Narrower means higher signal but fewer articles.
DEFAULT_NEWS_SOURCES = ["bloomberg.com", "reuters.com"]

# Optional historical range, as YYYY-MM-DD. Left as None so the default run fetches
# current news; set both (e.g. "2024-01-01" / "2024-12-31") to walk a past period in
# windows instead. Note that a backfill is one billed Valyu call per window per ticker.
START_DATE = None
END_DATE = None

# Days per request when a range is given. Valyu ranks by relevance rather than date, so
# a narrower window is the only way to get even coverage across a long period.
WINDOW_DAYS = 7

# Summarisation is off by default: it runs one model call per article, and FinBERT
# truncates to 512 tokens anyway, so for most articles it adds cost and a second
# model's bias without adding information.
SUMMARIZE = False
