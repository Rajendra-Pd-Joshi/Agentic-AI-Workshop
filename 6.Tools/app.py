import uuid

import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from backend import (
    chatbot,
    create_conversation,
    retrieve_all_threads,
    retrieve_thread_messages,
    touch_conversation,
    update_conversation_title,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Tool Metadata
# Maps each tool name → (emoji, human-readable label)
# ============================================================

TOOL_DISPLAY = {
    # ── Original tools ──────────────────────────────────────
    "get_current_datetime":      ("🕐", "Getting current date & time"),
    "calculate":                 ("🧮", "Running calculation"),
    "get_weather":               ("🌦️",  "Fetching live weather"),
    "search_wikipedia":          ("📖", "Searching Wikipedia"),
    "search_web":                ("🔍", "Searching the web"),
    "fetch_webpage":             ("🌐", "Fetching webpage content"),
    "convert_units":             ("📐", "Converting units"),
    "analyze_text":              ("📊", "Analysing text"),
    "generate_uuid":             ("🆔", "Generating UUID"),
    "random_number":             ("🎲", "Generating random number"),
    # ── Knowledge & Research ────────────────────────────────
    "search_arxiv":              ("📚", "Searching arXiv papers"),
    "search_pubmed":             ("🔬", "Searching PubMed"),
    "stackoverflow_search":      ("💻", "Searching Stack Overflow"),
    # ── Code & Developer ────────────────────────────────────
    "run_python":                ("🐍", "Running Python code"),
    "github_search":             ("🐙", "Searching GitHub"),
    "lint_code":                 ("🔎", "Linting code"),
    # ── Finance & Crypto ────────────────────────────────────
    "get_stock_price":           ("📈", "Fetching stock price"),
    "get_crypto_price":          ("₿",  "Fetching crypto price"),
    "get_forex_rate":            ("💱", "Fetching forex rate"),
    "portfolio_calculator":      ("💼", "Calculating portfolio value"),
    # ── File & Data ─────────────────────────────────────────
    "read_csv":                  ("📊", "Reading CSV file"),
    "extract_pdf_text":          ("📄", "Extracting PDF text"),
    "read_docx":                 ("📝", "Reading Word document"),
    "query_sql":                 ("🗄️",  "Running SQL query"),
    # ── Location & Maps ─────────────────────────────────────
    "geocode_address":           ("📍", "Geocoding address"),
    "get_timezone":              ("🕐", "Looking up timezone"),
    "ip_geolocation":            ("🌐", "Geolocating IP address"),
    # ── AI Memory ───────────────────────────────────────────
    "remember_fact":             ("🧠", "Storing memory"),
    "recall_memories":           ("🧠", "Recalling memories"),
    "forget_memory":             ("🗑️",  "Deleting memory"),
    "search_past_conversations": ("🔍", "Searching past conversations"),
    # ── Image ───────────────────────────────────────────────
    "resize_image":              ("🖼️",  "Resizing image"),
    "get_image_info":            ("🖼️",  "Reading image info"),
}

# Group tools for the sidebar reference panel
TOOL_GROUPS = {
    "🕐 Datetime & Utilities": [
        "get_current_datetime", "calculate", "convert_units",
        "analyze_text", "generate_uuid", "random_number",
    ],
    "🌦️ Weather & Location": [
        "get_weather", "geocode_address", "get_timezone", "ip_geolocation",
    ],
    "🔍 Search & Web": [
        "search_web", "fetch_webpage", "search_wikipedia",
    ],
    "📚 Knowledge & Research": [
        "search_arxiv", "search_pubmed", "stackoverflow_search",
    ],
    "💻 Code & Developer": [
        "run_python", "github_search", "lint_code",
    ],
    "💹 Finance": [
        "get_stock_price", "get_crypto_price", "get_forex_rate", "portfolio_calculator",
    ],
    "🗂️ File & Data": [
        "read_csv", "extract_pdf_text", "read_docx", "query_sql",
    ],
    "🖼️ Image": [
        "resize_image", "get_image_info",
    ],
    "🧠 AI Memory": [
        "remember_fact", "recall_memories", "forget_memory", "search_past_conversations",
    ],
}


# ============================================================
# Utility Functions
# ============================================================

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def generate_title(user_message: str) -> str:
    title = user_message.strip()
    if len(title) > 40:
        title = title[:40].rstrip() + "..."
    return title or "New Chat"


def get_tool_label(tool_name: str) -> tuple[str, str]:
    """Return (emoji, label) for a tool name, with a sensible fallback."""
    return TOOL_DISPLAY.get(tool_name, ("🔧", f"Using tool: {tool_name}"))


# ============================================================
# Session State
# ============================================================

def initialize_session_state():

    if "thread_id" not in st.session_state:
        conversations = retrieve_all_threads()
        if conversations:
            st.session_state.thread_id = conversations[0]["thread_id"]
        else:
            thread_id = generate_thread_id()
            create_conversation(thread_id, "New Chat")
            st.session_state.thread_id = thread_id

    if "message_history" not in st.session_state:
        st.session_state.message_history = retrieve_thread_messages(
            st.session_state.thread_id
        )


# ============================================================
# Create / Load Conversations
# ============================================================

def create_new_chat():
    thread_id = generate_thread_id()
    create_conversation(thread_id, "New Chat")
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []


def load_conversation(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = retrieve_thread_messages(thread_id)


# ============================================================
# Message Role Helper
# ============================================================

def get_message_role(message):
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    return None


# ============================================================
# Render Full Message History
# ============================================================

def render_messages():
    for message in st.session_state.message_history:

        role = get_message_role(message)

        # ── User message ─────────────────────────────────────
        if role == "user":
            content = message.content
            if not isinstance(content, str):
                content = str(content)
            with st.chat_message("user"):
                st.markdown(content)

        # ── AI message ───────────────────────────────────────
        elif role == "assistant":
            content = message.content
            has_text = (
                isinstance(content, str) and content.strip()
            ) or (
                isinstance(content, list)
                and any(
                    isinstance(block, dict) and block.get("type") == "text"
                    for block in content
                )
            )

            if not has_text:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        emoji, label = get_tool_label(tc.get("name", ""))
                        st.caption(f"{emoji} *{label}…*")
                continue

            if not isinstance(content, str):
                content = str(content)

            with st.chat_message("assistant"):
                st.markdown(content)

        # ── Tool result ──────────────────────────────────────
        elif role == "tool":
            tool_name = getattr(message, "name", "tool")
            emoji, label = get_tool_label(tool_name)
            result_content = message.content
            if not isinstance(result_content, str):
                result_content = str(result_content)

            with st.expander(f"{emoji} Tool result · `{tool_name}`", expanded=False):
                st.markdown(result_content)


# ============================================================
# Initialize
# ============================================================

initialize_session_state()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🤖 LangGraph Chatbot")

    # ── New chat button ──────────────────────────────────────
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()

    # ── Tool reference (grouped) ─────────────────────────────
    with st.expander("🛠️ Available Tools", expanded=False):
        for group_name, tool_names in TOOL_GROUPS.items():
            st.markdown(f"**{group_name}**")
            for name in tool_names:
                emoji, label = get_tool_label(name)
                st.markdown(f"&nbsp;&nbsp;&nbsp;{emoji} `{name}`")
            st.markdown("")

    st.divider()
    st.subheader("💬 Conversations")

    # ── Conversation list ────────────────────────────────────
    conversations = retrieve_all_threads()

    for conversation in conversations:
        thread_id = conversation["thread_id"]
        title = conversation["title"]
        is_current = thread_id == st.session_state.thread_id

        label = f"🟢 {title}" if is_current else f"💬 {title}"

        if st.button(label, key=f"conv_{thread_id}", use_container_width=True):
            if not is_current:
                load_conversation(thread_id)
                st.rerun()


# ============================================================
# Main Application
# ============================================================

st.title("🤖 LangGraph Chatbot")
st.caption(f"Thread ID: `{st.session_state.thread_id}`")
st.divider()


# ── Render existing messages ─────────────────────────────────
render_messages()


# ── Chat input ───────────────────────────────────────────────
user_input = st.chat_input("Type your message…")


# ============================================================
# Process User Input
# ============================================================

if user_input:

    is_first_message = len(st.session_state.message_history) == 0

    # ── Show user bubble immediately ─────────────────────────
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.message_history.append(
        HumanMessage(content=user_input)
    )

    # ── Auto-title on first message ──────────────────────────
    if is_first_message:
        title = generate_title(user_input)
        update_conversation_title(st.session_state.thread_id, title)

    # ── Tool status banner + AI response stream ───────────────
    tool_status = st.empty()
    ai_message = None
    active_tool_names: list[str] = []

    # Phase 1 — run graph via values stream to capture tool activity
    for event in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        stream_mode="values",
    ):
        msgs = event.get("messages", [])
        if not msgs:
            continue

        last_msg = msgs[-1]

        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls"):
            for tc in last_msg.tool_calls:
                name = tc.get("name", "tool")
                if name not in active_tool_names:
                    active_tool_names.append(name)
                    emoji, label = get_tool_label(name)
                    tool_status.info(f"{emoji} **{label}…**")

        elif isinstance(last_msg, ToolMessage):
            tool_status.empty()

    tool_status.empty()

    # Phase 2 — message stream for the final text response
    with st.chat_message("assistant"):

        try:
            def _text_stream():
                for chunk, meta in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                    stream_mode="messages",
                ):
                    if meta.get("langgraph_node") != "agent":
                        continue
                    content = chunk.content
                    if not content:
                        continue
                    yield content if isinstance(content, str) else str(content)

            ai_message = st.write_stream(_text_stream())

        except Exception as exc:
            st.error(f"Error generating response: {exc}")
            ai_message = None

    # ── Persist AI message to local history ──────────────────
    if ai_message:
        st.session_state.message_history.append(
            AIMessage(content=ai_message)
        )

    # ── Refresh tool result expanders ────────────────────────
    st.session_state.message_history = retrieve_thread_messages(
        st.session_state.thread_id
    )

    # ── Update conversation timestamp ────────────────────────
    touch_conversation(st.session_state.thread_id)

    st.rerun()