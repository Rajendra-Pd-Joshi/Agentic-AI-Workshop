import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from backend import chatbot


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# Utility Functions
# ============================================================

def generate_thread_id() -> str:
    """Generate a unique conversation/thread ID."""
    return str(uuid.uuid4())


def initialize_session_state():
    """Initialize all required Streamlit session state variables."""

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "thread_history" not in st.session_state:
        st.session_state.thread_history = []

    if "thread_id" not in st.session_state:
        thread_id = generate_thread_id()

        st.session_state.thread_id = thread_id
        st.session_state.thread_history.append(thread_id)


def create_new_chat():
    """Create and switch to a new conversation."""

    thread_id = generate_thread_id()

    st.session_state.thread_id = thread_id
    st.session_state.message_history = []

    if thread_id not in st.session_state.thread_history:
        st.session_state.thread_history.append(thread_id)


def load_conversation_history(thread_id: str):
    """
    Load messages belonging to a particular LangGraph thread
    and convert them into Streamlit's message format.
    """

    try:
        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        messages = state.values.get("messages", [])

        message_history = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            # Handle normal string content
            if isinstance(message.content, str):
                content = message.content

            # Handle structured content
            else:
                content = str(message.content)

            message_history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        st.session_state.thread_id = thread_id
        st.session_state.message_history = message_history

    except Exception as e:
        st.error(f"Unable to load conversation: {e}")


def render_message_history():
    """Render all previously stored messages."""

    for message in st.session_state.message_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])


def stream_ai_response(user_input: str):
    """
    Stream the assistant response from LangGraph.
    """

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    generator = chatbot.stream(
        input={
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config,
        stream_mode="messages",
    )

    for message_chunk, metadata in generator:

        content = message_chunk.content

        if not content:
            continue

        # Most chat models return a string.
        if isinstance(content, str):
            yield content

        # Some models can return structured content.
        else:
            yield str(content)


# ============================================================
# Initialize Session State
# ============================================================

initialize_session_state()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🤖 LangGraph Chatbot")

    # New Chat button
    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):
        create_new_chat()
        st.rerun()

    st.divider()

    st.subheader("💬 My Conversations")

    # Display newest conversations first
    for thread_id in reversed(st.session_state.thread_history):

        is_current_thread = (
            thread_id == st.session_state.thread_id
        )

        button_label = (
            f"🟢 {thread_id}"
            if is_current_thread
            else thread_id
        )

        if st.button(
            button_label,
            key=f"thread_{thread_id}",
            use_container_width=True,
        ):

            if thread_id != st.session_state.thread_id:
                load_conversation_history(thread_id)
                st.rerun()


# ============================================================
# Main UI
# ============================================================

st.title("LangGraph Chatbot")

st.caption(
    f"Conversation: {st.session_state.thread_id}"
)

st.divider()


# ============================================================
# Display Existing Messages
# ============================================================

render_message_history()


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input(
    "Type your message..."
)


# ============================================================
# Process User Message
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.write(user_input)

    # Save user message locally
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            ai_message = st.write_stream(
                stream_ai_response(user_input)
            )

        except Exception as e:

            st.error(
                f"Something went wrong while generating "
                f"the response: {e}"
            )

            ai_message = None

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    if ai_message:

        st.session_state.message_history.append(
            {
                "role": "assistant",
                "content": ai_message,
            }
        )

