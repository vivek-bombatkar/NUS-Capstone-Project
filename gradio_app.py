import re
from typing import Any

MAX_MEMORY = 5


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


def chat_fn(message: str, history: list[Any]) -> str:
    """Phase 3: route query with conversation context and normalized outputs."""
    context = _build_context_from_history(history, MAX_MEMORY)
    response = _route_query(message, context)
    return _format_router_output(response)


def launch_app() -> None:
    """Launch the Gradio ChatInterface."""
    import gradio as gr

    gr.ChatInterface(
        fn=chat_fn,
        title="NUS-Capstone-Project | CreativeFeedback AI ",
        description="Gradio frontend with context-aware multi-turn routing",
    ).launch()


if __name__ == "__main__":
    launch_app()
