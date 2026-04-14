from utils.llm import generate_response


# Static document store (acts as knowledge base)
DOCUMENTS = [
"""
Product Feedback Report Q1:
Many customers reported delays in shipping due to logistics issues.
Delivery timelines exceeded expectations in 40% of cases.
""",

"""
Customer Complaint Analysis:
Product defects were observed in multiple categories.
Customers reported damaged items and poor quality.
""",

"""
Customer Support Report:
Support response time is slow.
Average resolution time exceeds 48 hours.
""",

"""
Packaging Review:
Packaging quality is inconsistent.
Items often arrive damaged due to poor packaging.
"""
]


def retrieve_relevant_docs(query: str):
    """
    Use LLM to select relevant documents
    """

    prompt = f"""
You are a retrieval system.

User query:
{query}

Documents:
{DOCUMENTS}

Task:
- Select the most relevant 2 documents
- Return them exactly as they are
- Do NOT modify content
"""

    response = generate_response(prompt)

    return response


def run(query: str, context: str) -> str:
    """
    RAG Agent using LLM-based retrieval
    """

    retrieved_docs = retrieve_relevant_docs(query)

    prompt = f"""
You are an AI assistant performing document-based question answering.

User Question:
{query}

Relevant Documents:
{retrieved_docs}

Instructions:
- Answer ONLY using the provided documents
- Do NOT hallucinate
- Be concise
- Provide business insights

Answer:
"""

    answer = generate_response(prompt)

    return f"📄 Document-Based Answer (RAG):\n\n{answer}"