from transformers import pipeline
from config import SUMMARIZER_MODEL

summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)


# Summarize text using BART. Returns the original if it's too short or summarization fails.
def summarize_text(text: str) -> str:
    word_count = len(text.split())

    if word_count < 50:
        return text

    try:
        # Cap max_length below input length to avoid HuggingFace warnings
        max_len = min(60, int(word_count * 0.8))
        min_len = min(20, int(max_len * 0.5))

        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return text