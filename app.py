import streamlit as st
from controller.router import handle_query
from utils.db import create_table, insert_feedback

create_table()

# ----------------------------
# Config
# ----------------------------
MAX_MEMORY = 5  # number of recent interactions to keep

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
# Memory Management Functions
# ----------------------------
def add_message(role, content):
    """Add message and maintain memory limit"""
    st.session_state.messages.append({
        "role": role,
        "content": content
    })

    # Keep only last N messages
    if len(st.session_state.messages) > MAX_MEMORY * 2:
        st.session_state.messages = st.session_state.messages[-MAX_MEMORY * 2:]


def get_conversation_context():
    """Return memory as formatted text for LLM"""
    context = ""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        context += f"{role.upper()}: {content}\n"
    return context

# ----------------------------
# Display Chat History
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# User Input
# ----------------------------
user_input = st.chat_input("Ask about customer feedback...")

if user_input:
    # Add user message
    add_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get memory context (for future LLM use)
    context = get_conversation_context()


    response = handle_query(user_input, context)

    # Add assistant response
    add_message("user", user_input)

    # Store in DB
    insert_feedback(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)