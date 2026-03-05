# Financial Sentiment Analysis & News Summarization Pipeline

A modular Python tool that fetches financial news and earnings reports, summarizes long articles using **BART**, and performs sentiment analysis using **FinBERT** (Financial BERT).

## Features

- **Automated Data Fetching**: Retrieves the latest news (Bloomberg, Reuters) and earnings reports via the [Valyu API](https://valyu.ai/).
- **Intelligent Summarization**: Uses Facebook's `BART-large-cnn` to condense long financial articles into concise summaries, focusing on relevant snippets.
- **Financial Sentiment Analysis**: Employs `FinBERT`, a BERT model specialized for financial text, to score news as positive, neutral, or negative.
- **Modular Data Processing**: Clean project structure with separated concerns for fetching, processing, and storage.

## Project Structure

```text
├── fetcher/          # API clients and data fetching logic
├── processors/       # BART summarization and FinBERT sentiment scoring
├── utils/            # Shared utilities (logging, etc.)
├── main.py           # Main entry point for the pipeline
└── requirements.txt  # Project dependencies
```

## Prerequisites

To run this project, you will need:
- Python 3.8+
- A [Valyu API Key](https://valyu.ai/)

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LCS3002/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory (using `.env.example` as a template) and add your Valyu API key:
   ```env
   VALYU_API_KEY="your_api_key_here"
   ```

## Usage

Run the pipeline for a specific company by passing its ticker symbol to `main.py`:

```python
# Example in main.py
run("AAPL")
```

The script will:
1. Fetch latest earnings reports for Apple.
2. Fetch recent Bloomberg and Reuters news.
3. Filter and summarize the text.
4. Output a combined sentiment score and per-article results.

## Tech Stack

- **Language**: Python
- **APIs**: Valyu (Financial Data)
- **Deep Learning**: HuggingFace Transformers (BART & FinBERT), PyTorch
- **Data Handling**: Pandas

## License

This project is licensed under the MIT License - see the LICENSE file for details.
