import re
import streamlit as st
from controller.router import handle_query
from utils.db import create_table, insert_feedback
from utils.document_parser import parse_uploaded_file

create_table()

# ----------------------------
# Config
# ----------------------------
MAX_MEMORY = 5

st.set_page_config(
    page_title="CreativeFeedback AI",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# Sidebar — Document Upload
# ----------------------------
with st.sidebar:
    st.header("📂 Document Upload")
    st.caption("Upload files to query with RAG. Supports .txt and .pdf")

    uploaded_files = st.file_uploader(
        label="Upload documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Parse and store uploaded docs in session state
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []

    if uploaded_files:
        parsed = [parse_uploaded_file(f) for f in uploaded_files]
        st.session_state.uploaded_docs = parsed
        st.success(f"✅ {len(parsed)} document(s) loaded")
        for name, content in parsed:
            with st.expander(f"📄 {name}", expanded=False):
                preview = content[:500] + "..." if len(content) > 500 else content
                st.caption(preview)
    else:
        st.session_state.uploaded_docs = []
        st.info("No documents uploaded. Using built-in knowledge base.")

    st.divider()
    st.caption("💡 After uploading, ask questions like:\n- 'Summarise this document'\n- 'What does the report say about returns?'\n- 'What are the key issues?'")

# ----------------------------
# Main chat area
# ----------------------------
st.title("🤖 CreativeFeedback AI")
st.caption("Multi-Agent Feedback Intelligence Assistant")

# ----------------------------
# Session state init
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Memory management
# ----------------------------
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MEMORY * 2:
        st.session_state.messages = st.session_state.messages[-MAX_MEMORY * 2:]

def get_conversation_context():
    context = ""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        # RAG results are dicts — serialise just the answer for context
        if isinstance(content, dict):
            content = content.get("answer", "")
        context += f"{role.upper()}: {content}\n"
    return context

# ----------------------------
# RAG pipeline renderer
# ----------------------------
def render_rag_response(result: dict):
    # Show badge if answer came from uploaded docs
    if result.get("used_uploaded_docs"):
        names = ", ".join(result.get("uploaded_doc_names", []))
        st.info(f"📎 Answering from your uploaded document(s): **{names}**")
    else:
        st.info("📚 Answering from built-in knowledge base")

    with st.expander("🔍 Step 1: Query Expansion", expanded=False):
        st.markdown("Original query rewritten into alternative phrasings to improve retrieval recall:")
        for i, variant in enumerate(result["query_variants"]):
            prefix = "📝 Original" if i == 0 else f"🔄 Variant {i}"
            st.markdown(f"- **{prefix}:** {variant}")

    with st.expander("📊 Step 2: TF-IDF Retrieval + Similarity Scoring", expanded=False):
        if result["retrieved"]:
            validated_keys = {(s, c) for s, c, _ in result["validated"]}
            for source, chunk, score in result["retrieved"]:
                passed = (source, chunk) in validated_keys
                status = "✅ Passed re-ranking" if passed else "❌ Filtered by re-ranker"
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**📄 {source}**")
                    st.caption(chunk)
                with col2:
                    st.metric("Score", score)
                    st.markdown(status)
        else:
            st.warning("No chunks cleared the similarity threshold.")

    with st.expander("🧠 Step 3: LLM Re-ranking Validation", expanded=False):
        if result["validated"]:
            st.success(
                f"{len(result['validated'])} of {len(result['retrieved'])} "
                "chunks passed LLM relevance validation."
            )
            for source, chunk, score in result["validated"]:
                st.markdown(f"**✅ {source}** (score: {score})")
                st.caption(chunk)
        elif result["retrieved"]:
            st.error("All retrieved chunks were rejected by the LLM re-ranker.")
        else:
            st.warning("No chunks to re-rank.")

    st.markdown("---")
    if result["error"]:
        st.warning(f"📄 {result['error']}")
    else:
        st.markdown(f"📄 **Document-Based Answer (RAG):**\n\n{result['answer']}")
        st.caption(f"📎 Sources: {result['sources_cited']}")

# ----------------------------
# Render chat history
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, dict) and content.get("type") == "rag":
            render_rag_response(content)
        else:
            match = re.search(r"IMAGE_PATH::(.*)", str(content)) if content else None
            if match:
                st.image(match.group(1).strip(), caption="Generated Image", use_container_width=True)
            else:
                st.markdown(content)

# ----------------------------
# User input
# ----------------------------
user_input = st.chat_input("Ask about customer feedback, or query your uploaded documents...")

if user_input:
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = get_conversation_context()

    # Pass uploaded docs to the router
    response = handle_query(
        user_input,
        context,
        uploaded_docs=st.session_state.uploaded_docs or None
    )

    with st.chat_message("assistant"):
        if isinstance(response, dict) and response.get("type") == "rag":
            render_rag_response(response)
        else:
            match = re.search(r"IMAGE_PATH::(.*)", str(response)) if response else None
            if match:
                st.image(match.group(1).strip(), caption="Generated Image", use_container_width=True)
            else:
                st.markdown(response)

    # Fix memory bug — store assistant response, not duplicate user message
    add_message("assistant", response)
    insert_feedback(user_input)