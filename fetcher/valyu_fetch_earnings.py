from .valyu_client_class import ValyuClient


# Fetches earnings reports for given tickers via the Valyu API.
class EarningsFetcher:
    def __init__(self, tickers):
        self.client = ValyuClient()
        self.tickers = tickers

    def fetch(self):
        all_results = []
        for ticker in self.tickers:
            query = f"recent earnings reports about {ticker}"
            results = self.client.search(
                query,
                included_sources=["valyu/valyu-earnings-US"],
                max_results=10
            )
            all_results.extend(results)

        print(f"Total results: {len(all_results)}")
        if all_results:
            print(getattr(all_results[0], "title", str(all_results[0])))
        else:
            print("No earnings results found.")
        print("-" * 20)

        return all_results
