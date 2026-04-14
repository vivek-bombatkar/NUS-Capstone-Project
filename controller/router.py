from typing import List

# Import agents
from agents import sentiment_agent, trend_agent, recommendation_agent
from agents import sql_agent, rag_agent, image_agent
from utils.formatter import format_response

from utils.llm import generate_response
import json


def classify_query(query: str) -> List[str]:
    """
    LLM-based query classification for agent routing
    """

    prompt = f"""
You are an AI router for a multi-agent system.

Your job is to analyze the user query and decide which agents should be used.

Available agents:
- sentiment → analyze customer sentiment
- trend → detect recurring issues and patterns
- recommendation → suggest improvements
- sql → query structured data
- rag → retrieve knowledge from documents
- image → generate visualizations
- general → fallback if nothing matches

Instructions:
- Return ONLY a JSON list of agent names
- Use multiple agents if needed
- Be accurate and minimal (do not include unnecessary agents)

Examples:

Query: "Why are customers unhappy?"
Output: ["sentiment", "trend"]

Query: "Show customer data"
Output: ["sql"]

Query: "Summarize this document"
Output: ["rag"]

Query: "Why are customers unhappy and what should I do?"
Output: ["sentiment", "trend", "recommendation"]

Query: "{query}"
Output:
"""

    response = generate_response(prompt)

    try:
        agents = json.loads(response)

        if not isinstance(agents, list):
            raise ValueError("Invalid format")

        return agents

    except Exception:
        # Fallback (VERY IMPORTANT for robustness)
        return ["general"]
    

def handle_query(query: str, context: str) -> str:
    agents = classify_query(query)

    responses = []

    sentiment_result = None
    trend_result = None

    # ----------------------------
    # Step 1: Sentiment
    # ----------------------------
    if "sentiment" in agents:
        sentiment_result = sentiment_agent.run(query, context)
        responses.append(sentiment_result)

    # ----------------------------
    # Step 2: Trend
    # ----------------------------
    if "trend" in agents:
        trend_result = trend_agent.run(query, context)
        responses.append(trend_result)

    # ----------------------------
    # Step 3: Recommendation (uses previous results)
    # ----------------------------
    if "recommendation" in agents:
        combined_context = f"""
        {context}

        Sentiment Result:
        {sentiment_result}

        Trend Result:
        {trend_result}
        """
        rec_result = recommendation_agent.run(query, combined_context)
        responses.append(rec_result)

    # ----------------------------
    # Independent agents
    # ----------------------------
    if "sql" in agents:
        responses.append(sql_agent.run(query, context))

    if "rag" in agents:
        responses.append(rag_agent.run(query, context))

    if "image" in agents:
        image_result = image_agent.run(query, context)
        responses.append(image_result)

    if "general" in agents:
        responses.append(f"🤖 General Response:\n{context}")

    return format_response(responses)