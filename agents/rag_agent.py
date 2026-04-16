from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.llm import generate_response
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = "data/documents"
SIMILARITY_THRESHOLD = 0.05   # T1-2: minimum score to consider a chunk relevant
TOP_N = 2                      # number of chunks to retrieve


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_documents(directory: str = DOCUMENTS_DIR) -> list:
    """Load all .txt files from the documents directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename)) as f:
                documents.append((filename, f.read()))
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_documents(documents: list, chunk_size: int = 2) -> list:
    """
    Split each document into smaller chunks (by sentence groups).
    Returns a list of (source_name, chunk_text) tuples.
    """
    chunks = []
    for source_name, content in documents:
        sentences = [s.strip() for s in content.strip().split("\n") if s.strip()]
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i : i + chunk_size])
            chunks.append((source_name, chunk_text))
    return chunks


# ---------------------------------------------------------------------------
# T1-4: Query expansion — rewrite query into multiple phrasings
# ---------------------------------------------------------------------------
def expand_query(query: str) -> list:
    """
    Ask the LLM to generate 2 alternative phrasings of the query.
    Returns a list of query strings: [original] + [expansions].
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

    # Parse the numbered list — extract lines starting with 1. or 2.
    alternatives = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("1.") or line.startswith("2."):
            alternatives.append(line[2:].strip())

    # Always include the original; fall back gracefully if parsing fails
    return [query] + alternatives if alternatives else [query]


# ---------------------------------------------------------------------------
# T1-2 + T1-4: Retrieval with query expansion and similarity threshold
# ---------------------------------------------------------------------------
def retrieve_relevant_docs(query: str, top_n: int = TOP_N) -> list:
    """
    T1-4: Expand query into multiple phrasings, retrieve chunks for each,
    take the union of results.
    T1-2: Filter out chunks below the similarity threshold.
    Returns a list of (source_name, chunk_text, score) tuples,
    or an empty list if nothing clears the threshold.
    """
    documents = load_documents()
    chunks = chunk_documents(documents)

    if not chunks:
        return []

    chunk_texts = [text for _, text in chunks]
    chunk_sources = [source for source, _ in chunks]

    # T1-4: expand the query
    query_variants = expand_query(query)

    # Collect (index, best_score) across all query variants
    best_scores = {}   # chunk_index -> highest score seen across variants

    vectorizer = TfidfVectorizer()
    # Fit on all chunks + all query variants together for a consistent vocabulary
    vectorizer.fit(chunk_texts + query_variants)

    chunk_vectors = vectorizer.transform(chunk_texts)

    for variant in query_variants:
        query_vector = vectorizer.transform([variant])
        scores = cosine_similarity(query_vector, chunk_vectors).flatten()
        for idx, score in enumerate(scores):
            if idx not in best_scores or score > best_scores[idx]:
                best_scores[idx] = score

    # Sort by best score descending, take top_n
    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # T1-2: apply similarity threshold — discard chunks below the minimum
    retrieved = [
        (chunk_sources[idx], chunk_texts[idx], round(score, 4))
        for idx, score in ranked
        if score >= SIMILARITY_THRESHOLD
    ]

    return retrieved


# ---------------------------------------------------------------------------
# T1-3: Re-ranking — LLM validates whether retrieved chunks are actually relevant
# ---------------------------------------------------------------------------
def rerank_chunks(query: str, retrieved: list) -> list:
    """
    T1-3: Ask the LLM to validate each retrieved chunk for relevance.
    Filters out chunks the LLM considers irrelevant.
    Returns the filtered list of (source_name, chunk_text, score) tuples.
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

        # Accept the chunk only if the LLM says yes
        if verdict.startswith("y"):
            validated.append((source_name, chunk_text, score))

    return validated


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------
def run(query: str, context: str) -> dict:
    """
    RAG Agent — returns a structured dict so the UI can render
    each pipeline step independently.

    Returns:
    {
        "type": "rag",
        "query_variants": [...],
        "retrieved": [(source, chunk, score), ...],
        "validated": [(source, chunk, score), ...],
        "answer": "...",
        "sources_cited": "...",
        "error": None  # or an error message string
    }
    """

    result = {
        "type": "rag",
        "query_variants": [],
        "retrieved": [],
        "validated": [],
        "answer": "",
        "sources_cited": "",
        "error": None,
    }

    # --- Step 1a: Query expansion ---
    query_variants = expand_query(query)
    result["query_variants"] = query_variants

    # --- Step 1b+c: Retrieve with threshold ---
    retrieved = retrieve_relevant_docs(query, top_n=TOP_N)
    result["retrieved"] = retrieved

    if not retrieved:
        result["error"] = (
            "No relevant documents found. "
            "Try asking about delivery, support response times, or packaging."
        )
        return result

    # --- Step 2: Re-rank ---
    validated = rerank_chunks(query, retrieved)
    result["validated"] = validated

    if not validated:
        result["error"] = (
            "Documents were retrieved but none passed relevance validation. "
            "Please try a more specific query."
        )
        return result

    # --- Step 3: Generate ---
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