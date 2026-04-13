from typing import List

# Import agents
from agents import sentiment_agent, trend_agent, recommendation_agent
from agents import sql_agent, rag_agent, image_agent


def classify_query(query: str) -> List[str]:
    query = query.lower()
    agents = []

    if any(word in query for word in ["document", "pdf", "report", "summary"]):
        agents.append("rag")

    if any(word in query for word in ["data", "database", "records", "history"]):
        agents.append("sql")

    if any(word in query for word in ["sentiment", "feeling", "happy", "unhappy", "bad", "good"]):
        agents.append("sentiment")

    if any(word in query for word in ["trend", "pattern", "common", "issue", "problem"]):
        agents.append("trend")

    if any(word in query for word in ["recommend", "suggest", "improve", "solution"]):
        agents.append("recommendation")

    if any(word in query for word in ["image", "diagram", "visual", "chart"]):
        agents.append("image")

    if not agents:
        agents.append("general")

    return agents


def handle_query(query: str, context: str) -> str:
    agents = classify_query(query)

    responses = []

    for agent in agents:

        if agent == "rag":
            responses.append(rag_agent.run(query, context))

        elif agent == "sql":
            responses.append(sql_agent.run(query, context))

        elif agent == "sentiment":
            responses.append(sentiment_agent.run(query, context))

        elif agent == "trend":
            responses.append(trend_agent.run(query, context))

        elif agent == "recommendation":
            responses.append(recommendation_agent.run(query, context))

        elif agent == "image":
            responses.append(image_agent.run(query, context))

        elif agent == "general":
            responses.append(f"🤖 General Response using context:\n{context}")

    return "\n\n".join(responses)