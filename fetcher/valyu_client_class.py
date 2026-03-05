from valyu import Valyu


# Thin wrapper around the Valyu SDK with basic error handling.
class ValyuClient:
    def __init__(self):
        self.client = Valyu()

    def search(self, query, included_sources=None, max_results=10, start_date=None, end_date=None):
        try:
            response = self.client.search(
                query=query,
                included_sources=included_sources or [],
                max_num_results=max_results,
                start_date=start_date,
                end_date=end_date,
                search_type="all"
            )
            return response.results if hasattr(response, "results") else []

        except Exception as e:
            print(f"Error fetching data: {e}")
            return []
