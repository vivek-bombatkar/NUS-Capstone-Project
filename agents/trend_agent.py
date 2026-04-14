from utils.db import fetch_all_feedback
from collections import Counter


def run(query: str, context: str) -> str:
    """
    Trend Agent: Identify recurring issues in feedback
    """

    rows = fetch_all_feedback()

    if not rows:
        return "📈 No feedback data available for trend analysis."

    feedback_texts = [row[1].lower() for row in rows]

    # Define keyword categories
    issue_keywords = {
        "delivery": ["delivery", "shipping", "late", "delay"],
        "quality": ["quality", "broken", "defect", "damaged"],
        "packaging": ["packaging", "package"],
        "support": ["support", "service", "help"],
        "price": ["price", "cost", "expensive"],
    }

    issue_counts = Counter()

    # Count occurrences
    for text in feedback_texts:
        for category, keywords in issue_keywords.items():
            if any(keyword in text for keyword in keywords):
                issue_counts[category] += 1

    if not issue_counts:
        return "📈 No significant trends detected."

    # Get top issues
    top_issues = issue_counts.most_common(3)

    # Build response
    response = "📈 Trend Analysis:\n\nTop Customer Issues:\n"

    for issue, count in top_issues:
        response += f"- {issue.capitalize()}: {count} mentions\n"

    # Insight
    main_issue = top_issues[0][0]

    response += f"\n🔍 Key Insight:\nMost frequent issue is '{main_issue}'. Immediate attention recommended."

    return response