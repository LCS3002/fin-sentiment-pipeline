from .valyu_client_class import ValyuClient


# Fetches recent news articles for given tickers via the Valyu API.
class NewsFetcher:
    def __init__(self, tickers):
        self.client = ValyuClient()
        self.tickers = tickers

    def fetch(self):
        all_results = []
        for ticker in self.tickers:
            query = f"latest news about {ticker}"
            results = self.client.search(
                query,
                included_sources=["bloomberg.com", "reuters.com"],
                max_results=10
            )
            all_results.extend(results)

        print(f"Total results: {len(all_results)}")
        if all_results:
            print(getattr(all_results[0], "title", str(all_results[0])))
        else:
            print("No news results found.")
        print("-" * 20)

        return all_results
