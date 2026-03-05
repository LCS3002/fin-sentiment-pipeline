from dotenv import load_dotenv
import pandas as pd
import json
from fetcher import NewsFetcher, EarningsFetcher
from processors import SentimentScorer
from processors.BART_summarizer import summarize_text
from config import COMMON_NAMES

load_dotenv()


# Extract ticker, timestamp, and summarized text from raw API results into a DataFrame.
def panda_convert(results, ticker):
    rows = []
    company_name = COMMON_NAMES.get(ticker.upper(), ticker)

    for result in results:
        date = getattr(result, "publication_date", "No Information")

        # Flatten content if it comes back as a list of snippets
        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = " ".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content])

        full_text = f"{getattr(result, 'title', '')}. {content}"

        # Keep only sentences that mention the ticker or company name
        sentences = full_text.split(". ")
        relevant_sentences = [
            s for s in sentences
            if ticker.lower() in s.lower() or company_name.lower() in s.lower()
        ]

        filtered_text = ". ".join(relevant_sentences) if relevant_sentences else getattr(result, 'title', '')
        summary = summarize_text(filtered_text)

        rows.append({
            "ticker": ticker,
            "timestamp": date or "No Information",
            "clean_text": summary
        })

    print("\n--- ALL ROWS COLLECTED ---")
    for r in rows:
        print(json.dumps(r, indent=4))
        print("-" * 20)
    print("-" * 50)

    df = pd.DataFrame(rows, columns=["ticker", "timestamp", "clean_text"])
    return df


# Fetch news and earnings for a ticker, then run sentiment analysis.
def run(ticker):
    print("-" * 20)

    earnings = EarningsFetcher([ticker]).fetch()
    news = NewsFetcher([ticker]).fetch()

    df_news = panda_convert(news, ticker)
    news_score = SentimentScorer(df_news)

    print(news_score)


if __name__ == "__main__":
    ticker_input = input("Ticker: ")
    run(ticker_input)