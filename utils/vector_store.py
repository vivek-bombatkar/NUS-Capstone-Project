import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
embeddings = None


def add_documents(docs: list[str]):
    global documents, embeddings

    documents = docs
    embeddings = model.encode(docs)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def query_documents(query: str, k: int = 3):
    global embeddings, documents

    if embeddings is None:
        return ["No documents available"]

    query_embedding = model.encode([query])[0]

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in embeddings
    ]

    top_indices = np.argsort(similarities)[-k:][::-1]

    return [documents[i] for i in top_indices]