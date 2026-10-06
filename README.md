# Financial Sentiment Pipeline

Fetches financial news and earnings reports via the [Valyu API](https://valyu.ai/), scores
them with **FinBERT**, and persists the results to SQLite as a continuous per-article
sentiment signal.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FinBERT](https://img.shields.io/badge/model-ProsusAI%2Ffinbert-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What it produces

For each article: the full probability distribution over negative / neutral / positive,
plus a continuous **score = p_positive − p_negative** in [−1, 1].

```
ticker       date     score  articles
  AAPL 2024-03-05  0.914939         1
  AAPL 2024-03-06 -0.878249         1
```

The continuous score is the point. A discrete −1/0/+1 label throws away most of the
signal — a 0.51-confidence positive and a 0.99-confidence positive are not the same
observation, and anything downstream that aggregates sentiment needs the difference.

---

## Design notes

Three decisions worth explaining, because each one is a deliberate trade-off rather than
an omission.

**Publication dates are date-level at best, and sometimes absent.** Valyu's
`start_date` / `end_date` filters are `YYYY-MM-DD`, and `publication_date` comes back as
an unvalidated string that can be missing entirely. Unparseable and absent dates become
`NaT` and are **kept but excluded from daily aggregation** — an article with no date
cannot be aligned to a trading session, and quietly bucketing it into one would be worse
than dropping it. Anything consuming this for a backtest should assume an article dated
day *D* was only actionable from the open of *D+1*.

**Search is relevance-ranked, not chronological.** A single wide date range returns the N
most *relevant* hits in that window, not every article in it. Backfills therefore walk the
range in `WINDOW_DAYS` chunks and de-duplicate on url. Expect overlapping windows to
return the same article — that is normal, and the url primary key absorbs it.

**Summarisation is off by default.** BART runs one model call per article, which dominates
runtime on any sizeable fetch, and FinBERT truncates to 512 tokens anyway — so for most
articles summarising first costs time and injects a second model's bias without adding
information. Enable it with `--summarize` when articles are long and the lede matters.

### A caveat on FinBERT

`ProsusAI/finbert` is fine-tuned on the **Financial PhraseBank**. Evaluating it against
that benchmark therefore measures nothing useful, and its labels are a *feature*, not
ground truth. It is also easy to surprise — "Apple traded flat" scores as 92% negative.

---

## Project structure

```text
├── fetcher/
│   ├── valyu_client_class.py   # SDK wrapper: date plumbing, error handling, cost tracking
│   ├── valyu_fetch_news.py     # News, with windowed historical backfill
│   └── valyu_fetch_earnings.py # Valyu's proprietary US earnings dataset
├── processors/
│   ├── finBERT_sentiment.py    # Full probability distribution per article
│   └── BART_summarizer.py      # Optional summarisation (off by default)
├── storage/
│   └── db.py                   # SQLite store, upsert keyed on article url
├── utils/
│   └── logging.py              # Single logging setup
├── config.py                   # Models, sources, date range, window size
├── main.py                     # CLI entry point
└── requirements.txt
```

Both model wrappers load their weights on **first use**, not at import — importing the
package used to pull roughly 2 GB into memory whether or not anything was scored.

---

## Setup

Requires **Python 3.10+** and a [Valyu API key](https://valyu.ai/).

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add VALYU_API_KEY
```

---

## Usage

```bash
# current news for one ticker
python main.py AAPL

# historical backfill, walked in weekly windows
python main.py AAPL MSFT --start 2024-01-01 --end 2024-03-31

# include earnings reports, and summarise long articles first
python main.py AAPL --earnings --summarize
```

| Flag | Effect |
|---|---|
| `--start` / `--end` | `YYYY-MM-DD`; both required together. Enables windowed backfill. |
| `--earnings` | Also query Valyu's US earnings dataset |
| `--summarize` | Run BART over each article before scoring |
| `--db` | SQLite path (default `data/sentiment.db`) |

> A backfill is **one billed Valyu call per window per ticker**. A year of weekly windows
> for two tickers is ~104 calls. The client logs cumulative spend so it is visible as it
> runs.

Results are written to SQLite keyed on article url, so re-running over an overlapping
range updates rows rather than duplicating them.

```python
from storage import SentimentStore

store = SentimentStore()
store.daily_sentiment("AAPL")   # mean score per ticker-day, with article counts
store.load("AAPL", start="2024-01-01")
```

---

## Tech stack

Python · Valyu API · HuggingFace Transformers (FinBERT, BART) · PyTorch · pandas · SQLite

## License

MIT — see `LICENSE`.
