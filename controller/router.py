from typing import List
from agents import sentiment_agent, trend_agent, recommendation_agent
from agents import sql_agent, rag_agent, image_agent, weather_agent
from utils.formatter import format_response
from utils.llm import generate_response
import json


def classify_query(query: str) -> List[str]:
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
- Be accurate and minimal

Examples:
Query: "Why are customers unhappy?"
Output: ["sentiment", "trend"]

Query: "Show customer data"
Output: ["sql"]

Query: "Summarize this document"
Output: ["rag"]

Query: "What does the uploaded file say about returns?"
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


def handle_query(query: str, context: str, uploaded_docs: list = None):
    """
    Route query to agents and return response.
    uploaded_docs: optional list of (source_name, content) tuples from UI.

    Returns dict  when RAG is the sole agent (enables visual pipeline rendering).
    Returns str   in all other cases.
    """
    agents = classify_query(query)

    responses = []
    rag_result = None
    sentiment_result = None
    trend_result = None

    if "sentiment" in agents:
        sentiment_result = sentiment_agent.run(query, context)
        responses.append(sentiment_result)

    if "trend" in agents:
        trend_result = trend_agent.run(query, context)
        responses.append(trend_result)

    if "recommendation" in agents:
        combined_context = f"""
        {context}
        Sentiment Result: {sentiment_result}
        Trend Result: {trend_result}
        """
        responses.append(recommendation_agent.run(query, combined_context))

    if "sql" in agents:
        responses.append(sql_agent.run(query, context))

    if "rag" in agents:
        # Pass uploaded_docs through to the RAG agent
        rag_result = rag_agent.run(query, context, uploaded_docs=uploaded_docs)

    if "image" in agents:
        responses.append(image_agent.run(query, context))

    if "weather" in agents:
        responses.append(weather_agent.run(query, context))

    if "general" in agents:
        responses.append(f"🤖 General Response:\n{context}")

    # Return strategy
    if rag_result is not None and not responses:
        return rag_result  # dict — UI renders pipeline steps

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

    return format_response(responses)