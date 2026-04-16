import re
import streamlit as st
from controller.router import handle_query
from utils.db import create_table, insert_feedback

create_table()

# ----------------------------
# Config
# ----------------------------
MAX_MEMORY = 5

st.set_page_config(
    page_title="CreativeFeedback AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 CreativeFeedback AI")
st.caption("Multi-Agent Feedback Intelligence Assistant")

# ----------------------------
# Initialize Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Memory Management
# ----------------------------
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MEMORY * 2:
        st.session_state.messages = st.session_state.messages[-MAX_MEMORY * 2:]

def get_conversation_context():
    context = ""
    for msg in st.session_state.messages:
        context += f"{msg['role'].upper()}: {msg['content']}\n"
    return context

# ----------------------------
# RAG Pipeline Renderer
# ----------------------------
def render_rag_response(result: dict):
    """
    Renders each RAG pipeline step in a Streamlit expander,
    followed by the final answer inline.
    """

    # Step 1 — Query Expansion
    with st.expander("🔍 Step 1: Query Expansion", expanded=False):
        st.markdown("**Original query** was rewritten into alternative phrasings to improve retrieval recall:")
        for i, variant in enumerate(result["query_variants"]):
            prefix = "📝 Original" if i == 0 else f"🔄 Variant {i}"
            st.markdown(f"- **{prefix}:** {variant}")

    # Step 2 — TF-IDF Retrieval
    with st.expander("📊 Step 2: TF-IDF Retrieval + Similarity Scoring", expanded=False):
        if result["retrieved"]:
            st.markdown("**Chunks retrieved** and ranked by cosine similarity score:")
            for source, chunk, score in result["retrieved"]:
                passed = any(
                    s == source and c == chunk
                    for s, c, _ in result["validated"]
                )
                status = "✅ Passed re-ranking" if passed else "❌ Filtered by re-ranker"
                st.markdown(f"---")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**📄 {source}**")
                    st.caption(chunk)
                with col2:
                    st.metric("Score", score)
                    st.markdown(status)
        else:
            st.warning("No chunks cleared the similarity threshold.")

    # Step 3 — Re-ranking
    with st.expander("🧠 Step 3: LLM Re-ranking Validation", expanded=False):
        if result["validated"]:
            st.success(f"{len(result['validated'])} of {len(result['retrieved'])} chunks passed LLM relevance validation.")
            for source, chunk, score in result["validated"]:
                st.markdown(f"**✅ {source}** (score: {score})")
                st.caption(chunk)
        elif result["retrieved"]:
            st.error("All retrieved chunks were rejected by the LLM re-ranker.")
        else:
            st.warning("No chunks to re-rank — retrieval returned nothing.")

    # Final Answer
    st.markdown("---")
    if result["error"]:
        st.warning(f"📄 {result['error']}")
    else:
        st.markdown(f"📄 **Document-Based Answer (RAG):**\n\n{result['answer']}")
        st.caption(f"📎 Sources: {result['sources_cited']}")


# ----------------------------
# Render Chat History
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # RAG responses stored as dicts, others as plain strings
        if isinstance(message["content"], dict) and message["content"].get("type") == "rag":
            render_rag_response(message["content"])
        else:
            content = message["content"]
            match = re.search(r"IMAGE_PATH::(.*)", content) if isinstance(content, str) else None
            if match:
                st.image(match.group(1).strip(), caption="Generated Image", use_container_width=True)
            else:
                st.markdown(content)

# ----------------------------
# User Input
# ----------------------------
user_input = st.chat_input("Ask about customer feedback...")

if user_input:
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = get_conversation_context()
    response = handle_query(user_input, context)

    with st.chat_message("assistant"):
        if isinstance(response, dict) and response.get("type") == "rag":
            render_rag_response(response)
        else:
            match = re.search(r"IMAGE_PATH::(.*)", response) if isinstance(response, str) else None
            if match:
                st.image(match.group(1).strip(), caption="Generated Image", use_container_width=True)
            else:
                st.markdown(response)

    # Store response and fix the memory bug (assistant turn, not user)
    add_message("assistant", response)
    insert_feedback(user_input)