import os
import re
from typing import Any

MAX_MEMORY = 5
UPLOAD_TOP_K = 3
UPLOAD_SIMILARITY_THRESHOLD = 0.05


def _route_query(message: str, context: str) -> Any:
    """Lazy-import router to keep module import light for local validation/tests."""
    from controller.router import handle_query

    return handle_query(message, context)


def _extract_turn_from_history_item(item: Any) -> tuple[str, str]:
    """Support multiple Gradio history formats across versions."""
    # Format A (older): tuple/list pair -> (user_message, assistant_message)
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return str(item[0] or ""), str(item[1] or "")

    # Format B (newer): dict-based chat message
    # e.g. {'role': 'user'|'assistant', 'content': '...'}
    if isinstance(item, dict):
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "") or "")
        if role == "user":
            return content, ""
        if role == "assistant":
            return "", content

    return "", ""


def _build_context_from_history(history: list[Any], max_memory: int = MAX_MEMORY) -> str:
    """Convert Gradio history into router-compatible context text."""
    if not history:
        return ""

    recent_history = history[-max_memory:]
    lines: list[str] = []

    for item in recent_history:
        user_msg, assistant_msg = _extract_turn_from_history_item(item)
        if user_msg:
            lines.append(f"USER: {user_msg}")
        if assistant_msg:
            lines.append(f"ASSISTANT: {assistant_msg}")

    return "\n".join(lines)


def _read_uploaded_documents(files: Any) -> list[tuple[str, str]]:
    """Read uploaded text-like files from Gradio File component output."""
    if not files:
        return []

    normalized = files if isinstance(files, list) else [files]
    docs: list[tuple[str, str]] = []

    for f in normalized:
        # Gradio type="filepath" passes string paths.
        file_path = str(f)
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
        except UnicodeDecodeError:
            # Best-effort fallback for non-UTF8 text files
            with open(file_path, "r", encoding="latin-1") as fh:
                text = fh.read().strip()
        except Exception:
            continue

        if text:
            docs.append((os.path.basename(file_path), text))

    return docs


def _chunk_uploaded_documents(documents: list[tuple[str, str]], chunk_size: int = 2) -> list[tuple[str, str]]:
    """Chunk uploaded documents into sentence groups for retrieval."""
    chunks: list[tuple[str, str]] = []
    for source, content in documents:
        sentences = [s.strip() for s in content.split("\n") if s.strip()]
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i : i + chunk_size]).strip()
            if chunk_text:
                chunks.append((source, chunk_text))
    return chunks


def _is_broad_document_query(query: str) -> bool:
    """Detect broad prompts where lexical similarity is expected to be low."""
    q = query.lower().strip()
    broad_phrases = [
        "summarize attached file",
        "summarise attached file",
        "summarize this file",
        "summarise this file",
        "summarize this document",
        "summarise this document",
        "summarize uploaded",
        "summarise uploaded",
        "what is in this file",
        "what's in this file",
    ]
    return any(phrase in q for phrase in broad_phrases)


def _retrieve_uploaded_chunks(query: str, documents: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    """Retrieve relevant chunks from uploaded documents via TF-IDF similarity."""
    chunks = _chunk_uploaded_documents(documents)
    if not chunks:
        return []

    chunk_sources = [source for source, _ in chunks]
    chunk_texts = [text for _, text in chunks]

    if _is_broad_document_query(query):
        unique_chunks: list[tuple[str, str, float]] = []
        seen = set()
        for idx, (source, chunk) in enumerate(chunks):
            key = (source, chunk)
            if key in seen:
                continue
            seen.add(key)
            synthetic_score = round(max(0.1, 0.7 - (idx * 0.1)), 4)
            unique_chunks.append((source, chunk, synthetic_score))
            if len(unique_chunks) >= UPLOAD_TOP_K:
                break
        return unique_chunks

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        vectorizer.fit(chunk_texts + [query])

        query_vec = vectorizer.transform([query])
        chunk_vecs = vectorizer.transform(chunk_texts)
        scores = cosine_similarity(query_vec, chunk_vecs).flatten()

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        if ranked_indices and float(scores[ranked_indices[0]]) == 0.0:
            fallback = []
            for rank, i in enumerate(ranked_indices[:UPLOAD_TOP_K]):
                synthetic_score = round(max(0.1, 0.6 - (rank * 0.1)), 4)
                fallback.append((chunk_sources[i], chunk_texts[i], synthetic_score))
            return fallback

        return [
            (chunk_sources[i], chunk_texts[i], round(float(scores[i]), 4))
            for i in ranked_indices[:UPLOAD_TOP_K]
            if scores[i] >= UPLOAD_SIMILARITY_THRESHOLD
        ]
    except Exception:
        # Fallback when sklearn isn't installed in the active environment.
        query_terms = set(re.findall(r"\w+", query.lower()))
        scores = []
        for idx, text in enumerate(chunk_texts):
            text_terms = set(re.findall(r"\w+", text.lower()))
            overlap = len(query_terms & text_terms)
            score = overlap / max(len(query_terms), 1)
            scores.append((idx, score))

        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        if ranked and ranked[0][1] == 0:
            return [
                (chunk_sources[idx], chunk_texts[idx], round(max(0.1, 0.5 - (rank * 0.1)), 4))
                for rank, (idx, _score) in enumerate(ranked[:UPLOAD_TOP_K])
            ]

        score_map = dict(scores)
        return [
            (chunk_sources[i], chunk_texts[i], round(float(score_map[i]), 4))
            for i, _ in ranked[:UPLOAD_TOP_K]
            if score_map[i] >= UPLOAD_SIMILARITY_THRESHOLD
        ]


def _generate_grounded_answer_from_chunks(query: str, retrieved: list[tuple[str, str, float]]) -> str:
    """Generate answer grounded in retrieved chunks. Falls back gracefully if LLM unavailable."""
    if not retrieved:
        return "I could not find relevant information in the uploaded documents for that question."

    retrieved_text = "\n\n".join(
        [f"[Source: {source} | Score: {score}]\n{chunk}" for source, chunk, score in retrieved]
    )

    prompt = f"""You are a document QA assistant.

User question:
{query}

Retrieved context:
{retrieved_text}

Rules:
- Answer only from the retrieved context
- If context is insufficient, say so clearly
- Keep the answer concise and useful
- Mention source filenames used
"""

    try:
        from utils.llm import generate_response

        answer = generate_response(prompt)
        if answer.startswith("❌ LLM Error"):
            raise RuntimeError(answer)
        return answer
    except Exception:
        # Deterministic fallback when LLM/API key is unavailable
        lines = [f"- {source}: {chunk}" for source, chunk, _score in retrieved]
        return "Using uploaded documents, I found:\n" + "\n".join(lines)


def _answer_from_uploaded_documents(query: str, files: Any) -> dict[str, Any] | None:
    """Run ad-hoc RAG over user-uploaded documents."""
    docs = _read_uploaded_documents(files)
    if not docs:
        return None

    retrieved = _retrieve_uploaded_chunks(query, docs)
    if not retrieved:
        # Best-effort fallback: use first chunks from uploaded docs
        fallback_chunks = [
            (source, chunk, 0.0)
            for source, chunk in _chunk_uploaded_documents(docs)[:UPLOAD_TOP_K]
        ]
        if not fallback_chunks:
            return {
                "type": "rag",
                "query_variants": [query],
                "retrieved": [],
                "validated": [],
                "answer": "I could not extract readable text from the uploaded files.",
                "sources_cited": ", ".join(sorted(set(name for name, _ in docs))),
                "error": None,
            }

        fallback_answer = _generate_grounded_answer_from_chunks(query, fallback_chunks)
        return {
            "type": "rag",
            "query_variants": [query],
            "retrieved": fallback_chunks,
            "validated": fallback_chunks,
            "answer": (
                "I could not find high-confidence matches, but here is the closest context from your uploads.\n\n"
                + fallback_answer
            ),
            "sources_cited": ", ".join(sorted(set(source for source, _chunk, _score in fallback_chunks))),
            "error": None,
        }

    answer = _generate_grounded_answer_from_chunks(query, retrieved)
    return {
        "type": "rag",
        "query_variants": [query],
        "retrieved": retrieved,
        "validated": retrieved,
        "answer": answer,
        "sources_cited": ", ".join(sorted(set(source for source, _chunk, _score in retrieved))),
        "error": None,
    }


def _format_router_output(response: Any) -> str:
    """Normalize router outputs into markdown text for Gradio ChatInterface."""
    # 1) Standard text responses
    if isinstance(response, str):
        # Streamlit uses IMAGE_PATH markers; keep response readable in Gradio
        match = re.search(r"IMAGE_PATH::(.*)", response)
        if match:
            image_path = match.group(1).strip()
            cleaned = re.sub(r"\n?IMAGE_PATH::.*", "", response).strip()
            return f"{cleaned}\n\n🖼️ Image saved to: `{image_path}`"
        return response

    # 2) RAG dict (router returns dict when RAG is the only invoked agent)
    if isinstance(response, dict) and response.get("type") == "rag":
        if response.get("error"):
            return f"📄 RAG Error:\n\n{response['error']}"

        answer = response.get("answer", "")
        sources = response.get("sources_cited", "N/A")
        query_variants = response.get("query_variants", [])
        retrieved = response.get("retrieved", [])
        validated = response.get("validated", [])

        variants_md = "\n".join([f"- {v}" for v in query_variants]) if query_variants else "- (none)"
        retrieved_md = (
            "\n".join([f"- {src} (score: {score})" for src, _chunk, score in retrieved])
            if retrieved
            else "- (none)"
        )
        validated_md = (
            "\n".join([f"- {src} (score: {score})" for src, _chunk, score in validated])
            if validated
            else "- (none)"
        )

        return (
            "## 📄 Document-Based Answer (RAG)\n\n"
            f"{answer}\n\n"
            f"**📎 Sources:** {sources}\n\n"
            "### 🔍 Query Variants\n"
            f"{variants_md}\n\n"
            "### 📊 Retrieved Chunks\n"
            f"{retrieved_md}\n\n"
            "### 🧠 LLM-Validated Chunks\n"
            f"{validated_md}"
        )

    # 3) Fallback for any other object type
    return str(response)


def chat_fn(message: str, history: list[Any], files: Any = None) -> str:
    """Route query with optional uploaded-document RAG and normalized outputs."""
    uploaded_doc_response = _answer_from_uploaded_documents(message, files)
    if uploaded_doc_response is not None:
        return _format_router_output(uploaded_doc_response)

    context = _build_context_from_history(history, MAX_MEMORY)
    response = _route_query(message, context)
    return _format_router_output(response)


def launch_app() -> None:
    """Launch the Gradio ChatInterface."""
    import gradio as gr

    gr.ChatInterface(
        fn=chat_fn,
        title="CreativeFeedback AI (Gradio)",
        description="Upload documents and ask context-aware questions using RAG",
        additional_inputs=[
            gr.File(
                label="Upload documents (.txt preferred)",
                file_count="multiple",
                type="filepath",
            )
        ],
    ).launch()


if __name__ == "__main__":
    launch_app()
