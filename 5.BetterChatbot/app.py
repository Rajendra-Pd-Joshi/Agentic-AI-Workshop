
import uuid

import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from backend import (
    chatbot,
    create_conversation,
    retrieve_all_threads,
    retrieve_thread_messages,
    touch_conversation,
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
# Utility Functions
# ============================================================

def generate_thread_id() -> str:
    """
    Generate a unique thread ID.
    """

    return str(uuid.uuid4())


def generate_title(user_message: str) -> str:
    """
    Generate a simple conversation title from
    the user's first message.

    This does not call the LLM, so it is free.
    """

    title = user_message.strip()

    if len(title) > 40:
        title = title[:40].rstrip() + "..."

    return title or "New Chat"


# ============================================================
# Session State
# ============================================================

def initialize_session_state():

    if "thread_id" not in st.session_state:

        conversations = retrieve_all_threads()

        if conversations:

            st.session_state.thread_id = (
                conversations[0]["thread_id"]
            )

        else:

            thread_id = generate_thread_id()

            create_conversation(
                thread_id,
                "New Chat"
            )

            st.session_state.thread_id = thread_id


    if "message_history" not in st.session_state:

        st.session_state.message_history = (
            retrieve_thread_messages(
                st.session_state.thread_id
            )
        )


# ============================================================
# Create New Chat
# ============================================================

def create_new_chat():

    thread_id = generate_thread_id()

    create_conversation(
        thread_id,
        "New Chat"
    )

    st.session_state.thread_id = thread_id

    st.session_state.message_history = []


# ============================================================
# Load Conversation
# ============================================================

def load_conversation(thread_id: str):

    messages = retrieve_thread_messages(
        thread_id
    )

    st.session_state.thread_id = thread_id

    st.session_state.message_history = messages


# ============================================================
# Convert LangChain Message
# ============================================================

def get_message_role(message):

    if isinstance(message, HumanMessage):
        return "user"

    if isinstance(message, AIMessage):
        return "assistant"

    return None


# ============================================================
# Render Messages
# ============================================================

def render_messages():

    for message in st.session_state.message_history:

        role = get_message_role(message)

        if role is None:
            continue

        content = message.content

        if not isinstance(content, str):
            content = str(content)

        with st.chat_message(role):

            st.markdown(content)


# ============================================================
# Stream AI Response
# ============================================================

def stream_response(user_input: str):

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    generator = chatbot.stream(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ]
        },
        config=config,
        stream_mode="messages"
    )

    for message_chunk, metadata in generator:

        content = message_chunk.content

        if not content:
            continue

        if isinstance(content, str):

            yield content

        else:

            yield str(content)


# ============================================================
# Initialize
# ============================================================

initialize_session_state()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🤖 LangGraph Chatbot")

    # --------------------------------------------------------
    # New Chat
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()


    st.divider()

    st.subheader("💬 Conversations")


    # --------------------------------------------------------
    # Retrieve conversations
    # --------------------------------------------------------

    conversations = retrieve_all_threads()


    # --------------------------------------------------------
    # Display conversations
    # --------------------------------------------------------

    for conversation in conversations:

        thread_id = conversation["thread_id"]

        title = conversation["title"]


        is_current = (
            thread_id ==
            st.session_state.thread_id
        )


        if is_current:

            label = f"🟢 {title}"

        else:

            label = f"💬 {title}"


        if st.button(
            label,
            key=f"conversation_{thread_id}",
            use_container_width=True
        ):

            if not is_current:

                load_conversation(
                    thread_id
                )

                st.rerun()


# ============================================================
# Main Application
# ============================================================

st.title("🤖 LangGraph Chatbot")

st.caption(
    f"Thread ID: {st.session_state.thread_id}"
)


st.divider()


# ============================================================
# Existing Messages
# ============================================================

render_messages()


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input(
    "Type your message..."
)


# ============================================================
# Process User Input
# ============================================================

if user_input:

    # --------------------------------------------------------
    # First message?
    # --------------------------------------------------------

    is_first_message = (
        len(st.session_state.message_history) == 0
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------------------------
    # Store locally
    # --------------------------------------------------------

    st.session_state.message_history.append(
        HumanMessage(
            content=user_input
        )
    )


    # --------------------------------------------------------
    # Generate conversation title
    # --------------------------------------------------------

    if is_first_message:

        title = generate_title(
            user_input
        )

        from backend import update_conversation_title

        update_conversation_title(
            st.session_state.thread_id,
            title
        )


    # --------------------------------------------------------
    # Generate AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            ai_message = st.write_stream(
                stream_response(
                    user_input
                )
            )

        except Exception as e:

            st.error(
                f"Error generating response: {e}"
            )

            ai_message = None


    # --------------------------------------------------------
    # Store AI response
    # --------------------------------------------------------

    if ai_message:

        st.session_state.message_history.append(
            AIMessage(
                content=ai_message
            )
        )


    # --------------------------------------------------------
    # Update conversation timestamp
    # --------------------------------------------------------

    touch_conversation(
        st.session_state.thread_id
    )

