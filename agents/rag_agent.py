from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.llm import generate_response
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = "data/documents"
SIMILARITY_THRESHOLD = 0.05
TOP_N = 2


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_documents(
    directory: str = DOCUMENTS_DIR,
    uploaded_docs: list = None
) -> list:
    """
    Load documents from two sources and merge them:
    1. Static .txt files from data/documents/ (always loaded as baseline)
    2. Uploaded documents passed in from the UI session (optional)

    uploaded_docs: list of (source_name, content) tuples parsed from
                   Streamlit uploaded files via utils/document_parser.py

    Returns a deduplicated list of (source_name, content) tuples.
    Uploaded docs are prepended so they rank higher in retrieval when
    query terms match both sources equally.
    """
    static_docs = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, encoding="utf-8") as f:
                    static_docs.append((filename, f.read()))

    if uploaded_docs:
        # Uploaded docs go first so they are preferentially retrieved
        return uploaded_docs + static_docs

    return static_docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_documents(documents: list, chunk_size: int = 2) -> list:
    """
    Split each document into smaller chunks (by line groups).
    Returns a list of (source_name, chunk_text) tuples.
    """
    chunks = []
    for source_name, content in documents:
        sentences = [s.strip() for s in content.strip().split("\n") if s.strip()]
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i: i + chunk_size])
            chunks.append((source_name, chunk_text))
    return chunks


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
def expand_query(query: str) -> list:
    """
    Ask the LLM to generate 2 alternative phrasings of the query.
    Returns [original] + [alternatives].
    """
    prompt = f"""You are a search query assistant.

Original query: "{query}"

Rewrite this query into 2 alternative phrasings that mean the same thing
but use different words. This helps retrieve relevant documents that may
use different terminology.

Return ONLY the 2 alternatives as a plain numbered list, no explanation:
1.
2.
"""
    response = generate_response(prompt)
    alternatives = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("1.") or line.startswith("2."):
            alternatives.append(line[2:].strip())

    return [query] + alternatives if alternatives else [query]


# ---------------------------------------------------------------------------
# Retrieval with query expansion + similarity threshold
# ---------------------------------------------------------------------------
def retrieve_relevant_docs(
    query: str,
    top_n: int = TOP_N,
    uploaded_docs: list = None
) -> list:
    """
    Expand query → TF-IDF retrieval → similarity threshold filter.
    Returns list of (source_name, chunk_text, score) tuples.
    """
    documents = load_documents(uploaded_docs=uploaded_docs)
    chunks = chunk_documents(documents)

    if not chunks:
        return []

    chunk_texts = [text for _, text in chunks]
    chunk_sources = [source for source, _ in chunks]

    query_variants = expand_query(query)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(chunk_texts + query_variants)
    chunk_vectors = vectorizer.transform(chunk_texts)

    best_scores = {}
    for variant in query_variants:
        query_vector = vectorizer.transform([variant])
        scores = cosine_similarity(query_vector, chunk_vectors).flatten()
        for idx, score in enumerate(scores):
            if idx not in best_scores or score > best_scores[idx]:
                best_scores[idx] = score

    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    retrieved = [
        (chunk_sources[idx], chunk_texts[idx], round(score, 4))
        for idx, score in ranked
        if score >= SIMILARITY_THRESHOLD
    ]

    return retrieved


# ---------------------------------------------------------------------------
# Re-ranking — LLM validates chunk relevance
# ---------------------------------------------------------------------------
def rerank_chunks(query: str, retrieved: list) -> list:
    """
    Ask the LLM to validate each chunk for relevance.
    Returns only the chunks confirmed as relevant.
    """
    if not retrieved:
        return []

    validated = []
    for source_name, chunk_text, score in retrieved:
        prompt = f"""You are a relevance checker for a retrieval system.

Query: "{query}"

Retrieved chunk:
\"\"\"{chunk_text}\"\"\"

Is this chunk relevant to answering the query?
Reply with ONLY one word: Yes or No.
"""
        verdict = generate_response(prompt).strip().lower()
        if verdict.startswith("y"):
            validated.append((source_name, chunk_text, score))

    return validated


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------
def run(query: str, context: str, uploaded_docs: list = None) -> dict:
    """
    Full RAG pipeline:
      Step 1a — Query expansion
      Step 1b — TF-IDF retrieval across static + uploaded docs
      Step 1c — Similarity threshold filter
      Step 2  — LLM re-ranking validation
      Step 3  — Answer generation from validated chunks

    uploaded_docs: optional list of (source_name, content) tuples
                   injected from the Streamlit UI session.

    Returns a structured dict so app.py can render each step visually.
    """
    result = {
        "type": "rag",
        "query_variants": [],
        "retrieved": [],
        "validated": [],
        "answer": "",
        "sources_cited": "",
        "used_uploaded_docs": bool(uploaded_docs),
        "uploaded_doc_names": [name for name, _ in uploaded_docs] if uploaded_docs else [],
        "error": None,
    }

    # Step 1a: Query expansion
    query_variants = expand_query(query)
    result["query_variants"] = query_variants

    # Step 1b+c: Retrieve with threshold
    retrieved = retrieve_relevant_docs(query, top_n=TOP_N, uploaded_docs=uploaded_docs)
    result["retrieved"] = retrieved

    if not retrieved:
        result["error"] = (
            "No relevant content found. "
            "Try uploading a document or asking about delivery, support, or packaging."
        )
        return result

    # Step 2: Re-rank
    validated = rerank_chunks(query, retrieved)
    result["validated"] = validated

    if not validated:
        result["error"] = (
            "Documents were retrieved but none passed relevance validation. "
            "Please try a more specific query."
        )
        return result

    # Step 3: Generate
    retrieved_text = ""
    for source_name, chunk_text, score in validated:
        retrieved_text += f"[Source: {source_name} | Score: {score}]\n{chunk_text}\n\n"

    prompt = f"""You are an AI assistant performing document-based question answering.

User Question:
{query}

Retrieved Document Chunks:
{retrieved_text}

Instructions:
- Answer ONLY using the provided document chunks
- Do NOT hallucinate or use outside knowledge
- Be concise and provide business insights
- Reference the source name(s) in your answer

Answer:
"""

    answer = generate_response(prompt)
    result["answer"] = answer
    result["sources_cited"] = ", ".join(set(s for s, _, _sc in validated))

    return result