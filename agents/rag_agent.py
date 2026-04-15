from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.llm import generate_response


# Static document store (acts as knowledge base)
# Each entry is a (source_name, content) tuple so we can cite sources later
import os
DOCUMENTS = []
for filename in os.listdir("data/documents"):
    if filename.endswith(".txt"):
        with open(f"data/documents/{filename}") as f:
            DOCUMENTS.append((filename, f.read()))
            

def chunk_documents(documents: list, chunk_size: int = 2) -> list:
    """
    Split each document into smaller chunks (by sentence groups).
    Returns a list of (source_name, chunk_text) tuples.
    """
    chunks = []
    for source_name, content in documents:
        # Split on newlines, strip blanks
        sentences = [s.strip() for s in content.strip().split("\n") if s.strip()]
        # Group sentences into chunks of `chunk_size`
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i : i + chunk_size])
            chunks.append((source_name, chunk_text))
    return chunks


def retrieve_relevant_docs(query: str, top_n: int = 2) -> list:
    """
    Retrieve the top_n most relevant document chunks using
    TF-IDF vectorization + cosine similarity.
    Returns a list of (source_name, chunk_text) tuples.
    """
    chunks = chunk_documents(DOCUMENTS)
    chunk_texts = [text for _, text in chunks]
    chunk_sources = [source for source, _ in chunks]

    # Fit TF-IDF on all chunks plus the query
    vectorizer = TfidfVectorizer()
    all_texts = chunk_texts + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Query vector is the last row; chunk vectors are everything before it
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]

    # Compute cosine similarity between query and every chunk
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()

    # Pick top_n chunk indices by score
    top_indices = scores.argsort()[::-1][:top_n]

    retrieved = [(chunk_sources[i], chunk_texts[i]) for i in top_indices]
    return retrieved


def run(query: str, context: str) -> str:
    """
    RAG Agent: TF-IDF retrieval + LLM-based generation.
    Step 1 — RETRIEVE: find relevant chunks via cosine similarity
    Step 2 — GENERATE: answer using only the retrieved chunks
    """

    # --- Step 1: Retrieve ---
    retrieved = retrieve_relevant_docs(query, top_n=2)

    # Format retrieved chunks for the LLM prompt
    retrieved_text = ""
    for source_name, chunk_text in retrieved:
        retrieved_text += f"[Source: {source_name}]\n{chunk_text}\n\n"

    # --- Step 2: Generate ---
    prompt = f"""
You are an AI assistant performing document-based question answering.

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

    # Build cited sources string for display
    sources_cited = ", ".join(set(source for source, _ in retrieved))

    return (
        f"📄 Document-Based Answer (RAG):\n\n"
        f"{answer}\n\n"
        f"📎 Sources retrieved: {sources_cited}"
    )