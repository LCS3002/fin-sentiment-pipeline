# Ticker -> company name lookup for filtering relevant sentences
COMMON_NAMES = {
    "AAPL": "Apple",
    "META": "Meta",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon"
}

# Models
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
SENTIMENT_MODEL = "ProsusAI/finbert"

# Date range for fetching
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
