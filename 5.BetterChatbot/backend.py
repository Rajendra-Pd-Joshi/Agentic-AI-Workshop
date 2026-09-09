import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Database
# ============================================================

DB_NAME = "chatbot.db"

conn = sqlite3.connect(
    DB_NAME,
    check_same_thread=False
)


# ============================================================
# Conversation Metadata
# ============================================================

def initialize_database():
    """
    Create the conversations table if it does not exist.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()


initialize_database()


# ============================================================
# LangGraph State
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# Chat Node
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# Build Graph
# ============================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_edge(
    "chat_node",
    END
)


# ============================================================
# LangGraph Checkpointer
# ============================================================

checkpointer = SqliteSaver(conn)


# ============================================================
# Compile Chatbot
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# Conversation Functions
# ============================================================

def create_conversation(thread_id: str, title: str):
    """
    Create a new conversation in the metadata database.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO conversations
        (thread_id, title)
        VALUES (?, ?)
        """,
        (thread_id, title)
    )

    conn.commit()


def update_conversation_title(
    thread_id: str,
    title: str
):
    """
    Update the title of an existing conversation.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET title = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE thread_id = ?
        """,
        (title, thread_id)
    )

    conn.commit()


def touch_conversation(thread_id: str):
    """
    Update the last-used timestamp of a conversation.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    conn.commit()


def delete_conversation(thread_id: str):
    """
    Delete conversation metadata.

    Note:
    LangGraph checkpoints are stored separately.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM conversations
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    conn.commit()


def retrieve_all_threads():
    """
    Return all conversations ordered by most recently updated.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT thread_id, title, created_at, updated_at
        FROM conversations
        ORDER BY updated_at DESC
        """
    )

    rows = cursor.fetchall()

    conversations = []

    for row in rows:

        conversations.append(
            {
                "thread_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
        )

    return conversations


def retrieve_thread_messages(thread_id: str):
    """
    Retrieve messages for a specific LangGraph thread.
    """

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get(
        "messages",
        []
    )


def conversation_exists(thread_id: str) -> bool:
    """
    Check whether a conversation exists.
    """

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 1
        FROM conversations
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    return cursor.fetchone() is not None

