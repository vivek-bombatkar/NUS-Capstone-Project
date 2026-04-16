from typing import List

# Import agents
from agents import sentiment_agent, trend_agent, recommendation_agent
from agents import sql_agent, rag_agent, image_agent, weather_agent
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
- weather → provide weather information
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
        return ["general"]


def handle_query(query: str, context: str):
    """
    Route query to the appropriate agents and return results.

    Return type is intentionally flexible:
    - Returns a dict  when RAG is the sole agent (so the UI can render
      the pipeline steps visually)
    - Returns a str   in all other cases (plain text or formatted
      multi-agent response)
    """
    agents = classify_query(query)

    responses = []
    rag_result = None

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
        # Keep RAG result separate — it's a dict, not a string
        rag_result = rag_agent.run(query, context)

    if "image" in agents:
        responses.append(image_agent.run(query, context))

    if "weather" in agents:
        responses.append(weather_agent.run(query, context))

    if "general" in agents:
        responses.append(f"🤖 General Response:\n{context}")

    # ----------------------------
    # Return strategy
    # ----------------------------

    # Case 1: RAG is the only agent invoked → return the dict directly
    # so app.py can render the full pipeline visualisation
    if rag_result is not None and not responses:
        return rag_result

    # Case 2: RAG ran alongside other agents → extract the plain-text
    # answer from the dict and add it to the responses list as a string,
    # then format everything together normally
    if rag_result is not None and responses:
        if rag_result.get("error"):
            responses.append(f"📄 RAG: {rag_result['error']}")
        else:
            rag_text = (
                f"📄 Document-Based Answer (RAG):\n\n"
                f"{rag_result.get('answer', '')}\n\n"
                f"📎 Sources: {rag_result.get('sources_cited', '')}"
            )
            responses.append(rag_text)

    # Case 3: No RAG involved → format and return as normal
    return format_response(responses)