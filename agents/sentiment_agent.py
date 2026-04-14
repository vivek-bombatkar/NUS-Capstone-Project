from utils.db import fetch_all_feedback


def run(query: str, context: str) -> str:
    """
    Sentiment Agent: Analyze sentiment distribution from feedback
    """

    rows = fetch_all_feedback()

    if not rows:
        return "😊 No feedback data available for sentiment analysis."

    sentiments = [row[2] for row in rows]

    total = len(sentiments)
    positive = sentiments.count("positive")
    negative = sentiments.count("negative")
    neutral = sentiments.count("neutral")

    # Calculate percentages
    pos_pct = round((positive / total) * 100, 1)
    neg_pct = round((negative / total) * 100, 1)
    neu_pct = round((neutral / total) * 100, 1)

    # Insight generation
    if neg_pct > 50:
        insight = "⚠️ High customer dissatisfaction detected."
    elif pos_pct > 50:
        insight = "✅ Overall customer satisfaction is strong."
    else:
        insight = "⚖️ Customer sentiment is mixed."

    response = f"""
😊 Sentiment Analysis:

Total Feedback: {total}

Distribution:
- Positive: {positive} ({pos_pct}%)
- Negative: {negative} ({neg_pct}%)
- Neutral: {neutral} ({neu_pct}%)

Insight:
{insight}
"""

    return response.strip()