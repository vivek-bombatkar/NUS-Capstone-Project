from utils.llm import generate_response


def run(query: str, context: str) -> str:
    """
    Recommendation Agent: LLM-powered insights
    """

    # Build prompt
    prompt = f"""
You are a business AI assistant helping SMEs improve their products based on customer feedback.

User Question:
{query}

Context from previous analysis:
{context}

Your task:
- Analyze the sentiment and trends
- Identify key business problems
- Suggest 3-5 actionable recommendations

Rules:
- Be concise
- Be practical
- Use bullet points
- Focus on business impact

Output format:
💡 Recommendations:
- ...
- ...
"""

    response = generate_response(prompt)

    return response