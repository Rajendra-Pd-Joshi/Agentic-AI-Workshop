import hashlib
import html
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

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
    # RAG
    build_rag_index,
    list_rag_pdfs,
    clear_rag_index,
)


# ============================================================
# Configuration
# Requires Streamlit 1.40+ for keyed containers.
# ============================================================

APP_NAME = "LangGraph Chatbot"
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="auto",
)


# ============================================================
# Tool metadata
# ============================================================

TOOL_DISPLAY = {
    "get_current_datetime": ("🕐", "Getting current date & time"),
    "calculate": ("🧮", "Running calculation"),
    "get_weather": ("🌦️", "Fetching live weather"),
    "search_wikipedia": ("📖", "Searching Wikipedia"),
    "search_web": ("🔍", "Searching the web"),
    "get_news": ("📰", "Fetching latest news"),
    "fetch_webpage": ("🌐", "Fetching webpage content"),
    "convert_units": ("📐", "Converting units"),
    "analyze_text": ("📊", "Analysing text"),
    "generate_uuid": ("🆔", "Generating UUID"),
    "random_number": ("🎲", "Generating random number"),
    "search_arxiv": ("📚", "Searching arXiv papers"),
    "search_pubmed": ("🔬", "Searching PubMed"),
    "stackoverflow_search": ("💻", "Searching Stack Overflow"),
    "run_python": ("🐍", "Running Python code"),
    "github_search": ("🐙", "Searching GitHub"),
    "lint_code": ("🔎", "Linting code"),
    "get_stock_price": ("📈", "Fetching stock price"),
    "get_crypto_price": ("₿", "Fetching crypto price"),
    "get_forex_rate": ("💱", "Fetching forex rate"),
    "portfolio_calculator": ("💼", "Calculating portfolio value"),
    "read_csv": ("📊", "Reading CSV file"),
    "extract_pdf_text": ("📄", "Extracting PDF text"),
    "read_docx": ("📝", "Reading Word document"),
    "query_sql": ("🗄️", "Running SQL query"),
    "geocode_address": ("📍", "Geocoding address"),
    "get_timezone": ("🕐", "Looking up timezone"),
    "ip_geolocation": ("🌐", "Geolocating IP address"),
    "remember_fact": ("🧠", "Storing memory"),
    "recall_memories": ("🧠", "Recalling memories"),
    "forget_memory": ("🗑️", "Deleting memory"),
    "search_past_conversations": ("🔍", "Searching past conversations"),
    "resize_image": ("🖼️", "Resizing image"),
    "get_image_info": ("🖼️", "Reading image info"),
    "query_pdf": ("📄", "Searching PDF with MMR"),
}

TOOL_GROUPS = {
    "🕐 Datetime & Utilities": [
        "get_current_datetime",
        "calculate",
        "convert_units",
        "analyze_text",
        "generate_uuid",
        "random_number",
    ],
    "🌦️ Weather & Location": [
        "get_weather",
        "geocode_address",
        "get_timezone",
        "ip_geolocation",
    ],
    "🔍 Search & Web": [
        "search_web",
        "get_news",
        "fetch_webpage",
        "search_wikipedia",
    ],
    "📚 Knowledge & Research": [
        "search_arxiv",
        "search_pubmed",
        "stackoverflow_search",
    ],
    "💻 Code & Developer": [
        "run_python",
        "github_search",
        "lint_code",
    ],
    "💹 Finance": [
        "get_stock_price",
        "get_crypto_price",
        "get_forex_rate",
        "portfolio_calculator",
    ],
    "🗂️ File & Data": [
        "read_csv",
        "extract_pdf_text",
        "read_docx",
        "query_sql",
    ],
    "🖼️ Image": [
        "resize_image",
        "get_image_info",
    ],
    "🧠 AI Memory": [
        "remember_fact",
        "recall_memories",
        "forget_memory",
        "search_past_conversations",
    ],
    "📄 PDF / RAG": [
        "query_pdf",
    ],
}


# ============================================================
# Welcome suggestions
# ============================================================

SUGGESTIONS = [
    {
        "title": "Research a topic",
        "description": "Find and compare useful sources",
        "prompt": (
            "Help me research a topic. First ask what I want to "
            "investigate and how detailed the answer should be."
        ),
        "icon": "travel_explore",
    },
    {
        "title": "Analyze a document",
        "description": "Explore an indexed PDF",
        "prompt": (
            "Help me analyze a PDF. If this conversation has an indexed "
            "document, summarize its key ideas. Otherwise, explain how "
            "to upload and index one using the Documents control."
        ),
        "icon": "description",
    },
    {
        "title": "Solve a coding problem",
        "description": "Debug, explain, or improve code",
        "prompt": (
            "Help me solve a coding problem. Ask me to share the code, "
            "the expected behavior, and any error messages."
        ),
        "icon": "code",
    },
    {
        "title": "Create a plan",
        "description": "Turn an idea into practical steps",
        "prompt": (
            "Help me create an actionable plan. First ask about my goal, "
            "timeline, and important constraints."
        ),
        "icon": "checklist",
    },
]


# ============================================================
# CSS
# ============================================================

CSS = """
<style>
.stApp {
    --bg: var(--background-color, #ffffff);
    --sidebar-bg: var(--secondary-background-color, #f7f7f8);
    --text: var(--text-color, #202123);
    --muted: color-mix(in srgb, var(--text) 68%, var(--bg));
    --border: color-mix(in srgb, var(--text) 15%, var(--bg));
    --hover: color-mix(in srgb, var(--text) 7%, var(--bg));
    --user-bg: color-mix(in srgb, var(--text) 6%, var(--bg));
    --composer-bg: var(--bg);
    --accent: #168477;
    --focus: #168477;
    --chat-width: 800px;
    --radius: 12px;
    --ease: 170ms ease;
    background: var(--bg);
    color: var(--text);
    font-family:
        Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
        Roboto, Helvetica, Arial, sans-serif;
}

.stApp *,
.stApp *::before,
.stApp *::after {
    box-sizing: border-box;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg);
}

[data-testid="stHeader"] {
    background: var(--bg);
}

[data-testid="stMainBlockContainer"] {
    max-width: none;
    padding: 3.4rem 2rem 2rem;
}

.stApp p,
.stApp li,
.stApp textarea,
.stApp input {
    font-family:
        Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
        Roboto, Helvetica, Arial, sans-serif;
}

.stApp button {
    transition:
        background var(--ease),
        border-color var(--ease),
        box-shadow var(--ease),
        transform var(--ease);
}

.stApp button:focus-visible,
.stApp a:focus-visible,
.stApp input:focus-visible,
.stApp textarea:focus-visible,
.stApp summary:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 3px;
}

.stApp [data-testid="stButton"] button,
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stPopover"] button {
    border-radius: var(--radius);
    box-shadow: none;
}

[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebarContent"] {
    background: var(--sidebar-bg);
}

[data-testid="stSidebarUserContent"] {
    padding: 1.2rem 0.8rem 0.8rem;
}

.st-key-sidebar_shell {
    min-height: calc(100dvh - 6.2rem);
}

.st-key-sidebar_shell > [data-testid="stVerticalBlock"] {
    min-height: inherit;
    gap: 0.75rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    min-height: 40px;
    padding: 0 0.45rem;
    margin-bottom: 0.35rem;
}

.brand-mark,
.welcome-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: var(--accent);
}

.brand-mark {
    width: 30px;
    height: 30px;
}

.brand-name {
    font-size: 15px;
    font-weight: 650;
    letter-spacing: -0.3px;
}

.st-key-new_chat_action button {
    min-height: 44px;
    justify-content: flex-start;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
}

.st-key-new_chat_action button:hover {
    background: var(--hover);
    border-color: var(--border);
}

.st-key-conversation_search [data-baseweb="input"] {
    background: var(--hover);
    border: 1px solid transparent;
    border-radius: 10px;
}

.st-key-conversation_search input {
    font-size: 13px;
}

.nav-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.075em;
    text-transform: uppercase;
    padding: 0.6rem 0.6rem 0;
}

.st-key-conversation_list {
    max-height: max(160px, calc(100dvh - 420px));
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}

.st-key-conversation_list [data-testid="stVerticalBlock"] {
    gap: 0.2rem;
}

.st-key-conversation_list button {
    min-height: 40px;
    width: 100%;
    justify-content: flex-start;
    padding: 0.5rem 0.65rem;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text);
}

.st-key-conversation_list button:hover {
    background: var(--hover);
    border-color: transparent;
    color: var(--text);
}

.st-key-conversation_list button[kind="primary"] {
    background: var(--hover);
    border-color: transparent;
    color: var(--text);
    font-weight: 600;
}

.st-key-conversation_list button [data-testid="stMarkdownContainer"] {
    min-width: 0;
    overflow: hidden;
    text-align: left;
}

.st-key-conversation_list button p {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
}

.st-key-sidebar_footer {
    position: sticky;
    bottom: 0;
    margin-top: auto;
    padding-top: 0.8rem;
    background: var(--sidebar-bg);
    border-top: 1px solid var(--border);
    z-index: 2;
}

.st-key-sidebar_footer [data-testid="stPopover"] button {
    min-height: 40px;
    justify-content: flex-start;
    background: transparent;
    border-color: transparent;
}

.st-key-sidebar_footer [data-testid="stPopover"] button:hover {
    background: var(--hover);
}

.version-note {
    padding: 0.4rem 0.5rem;
    color: var(--muted);
    font-size: 11px;
}

.st-key-topbar {
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.55rem;
}

.st-key-topbar [data-testid="stHorizontalBlock"] {
    align-items: center;
    flex-wrap: nowrap;
}

.st-key-topbar [data-testid="stColumn"] {
    min-width: 0;
}

.topbar-title {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    height: 42px;
    min-width: 0;
}

.topbar-title .conversation-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 550;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.assistant-label {
    flex-shrink: 0;
    color: var(--muted);
    font-size: 12px;
    padding-right: 0.65rem;
    border-right: 1px solid var(--border);
}

.st-key-topbar button {
    min-height: 38px;
    background: transparent;
    border-color: transparent;
}

.st-key-topbar button:hover {
    background: var(--hover);
}

.st-key-conversation,
.st-key-welcome {
    width: 100%;
    max-width: var(--chat-width);
    margin-left: auto;
    margin-right: auto;
}

.st-key-conversation {
    padding-bottom: 1.5rem;
}

.st-key-conversation [data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 0.6rem 0;
    gap: 0.8rem;
    box-shadow: none;
    min-width: 0;
}

.st-key-conversation [data-testid="stChatMessageContent"] {
    min-width: 0;
    max-width: 100%;
}

.st-key-conversation [data-testid="stMarkdownContainer"] {
    font-size: 16px;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.st-key-conversation [data-testid="stMarkdownContainer"] p,
.st-key-conversation [data-testid="stMarkdownContainer"] li {
    font-size: 16px;
    line-height: 1.65;
}

.st-key-conversation [data-testid="stMarkdownContainer"] h1 {
    font-size: 1.55rem;
}

.st-key-conversation [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.3rem;
}

.st-key-conversation [data-testid="stMarkdownContainer"] h3 {
    font-size: 1.1rem;
}

.st-key-conversation [data-testid="stMarkdownContainer"] a {
    color: var(--accent);
    text-underline-offset: 3px;
}

[class*="st-key-user_message_"] {
    max-width: 75%;
    margin-left: auto;
    margin-top: 0.6rem;
    margin-bottom: 0.9rem;
}

[class*="st-key-user_message_"] [data-testid="stChatMessage"] {
    padding: 0.8rem 1.1rem;
    border-radius: 20px;
    background: var(--user-bg);
}

[class*="st-key-user_message_"] [data-testid="stChatMessageAvatarUser"] {
    display: none;
}

[class*="st-key-assistant_message_"] {
    margin-bottom: 0.9rem;
}

.st-key-conversation [data-testid="stChatMessageAvatarAssistant"] {
    width: 27px;
    height: 27px;
    min-width: 27px;
    background: transparent;
    color: var(--accent);
}

.st-key-conversation pre {
    border: 1px solid var(--border);
    border-radius: 12px;
    max-width: 100%;
    overflow-x: auto;
    white-space: pre;
}

.st-key-conversation code {
    font-family:
        "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.88em;
}

.st-key-conversation [data-testid="stTable"] {
    max-width: 100%;
    overflow-x: auto;
}

.st-key-conversation blockquote {
    border-left: 3px solid var(--border);
    color: var(--muted);
}

.st-key-conversation [data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

.st-key-welcome {
    padding-top: clamp(1rem, 10vh, 7rem);
    padding-bottom: 2rem;
}

.welcome-copy {
    text-align: center;
    margin: 0 auto 1.6rem;
    max-width: 570px;
}

.welcome-mark {
    width: 48px;
    height: 48px;
    margin-bottom: 0.75rem;
}

.welcome-copy h1 {
    color: var(--text);
    font-size: clamp(25px, 3vw, 34px);
    line-height: 1.2;
    letter-spacing: -1px;
    font-weight: 600;
    margin: 0 0 0.8rem;
    padding: 0;
}

.welcome-copy p {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
}

.st-key-suggestions button {
    min-height: 76px;
    padding: 0.85rem 1rem;
    justify-content: flex-start;
    text-align: left;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 14px;
}

.st-key-suggestions button:hover {
    background: var(--hover);
    border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
    transform: translateY(-2px);
}

.st-key-suggestions button p {
    text-align: left;
    font-size: 13px;
    line-height: 1.5;
}

.st-key-suggestions button strong {
    font-size: 14px;
    font-weight: 550;
}

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: var(--bg);
}

[data-testid="stBottomBlockContainer"] {
    max-width: 848px;
    margin-left: auto;
    margin-right: auto;
    padding: 0.65rem 1.5rem max(1rem, env(safe-area-inset-bottom));
}

[data-testid="stChatInput"] {
    background: var(--composer-bg);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: 0 4px 22px rgba(0, 0, 0, 0.055);
    padding: 0.3rem 0.5rem;
    transition: border-color var(--ease), box-shadow var(--ease);
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 12%, transparent);
}

[data-testid="stChatInput"] textarea {
    font-size: 16px;
    line-height: 1.5;
    color: var(--text);
}

[data-testid="stChatInputSubmitButton"] {
    border-radius: 50%;
    color: var(--accent);
}

.file-chip {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 0.75rem;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--hover);
}

.file-chip .file-name {
    min-width: 0;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
}

.file-chip .file-state {
    font-size: 11px;
    color: var(--muted);
    flex-shrink: 0;
}

.tool-group {
    font-size: 12px;
    font-weight: 600;
    margin: 0.7rem 0 0.35rem;
}

.tool-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}

.tool-pill {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background: var(--hover);
    border: 1px solid var(--border);
    border-radius: 7px;
    font-size: 11px;
    overflow-wrap: anywhere;
}

.activity {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-height: 26px;
    color: var(--muted);
    font-size: 13px;
    padding: 0.2rem 0;
}

.activity-dots {
    display: inline-flex;
    gap: 3px;
}

.activity-dots i {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: currentColor;
    animation: thinking 1.2s ease-in-out infinite;
}

.activity-dots i:nth-child(2) {
    animation-delay: 0.15s;
}

.activity-dots i:nth-child(3) {
    animation-delay: 0.3s;
}

.st-key-assistant_message_live {
    animation: response-enter 180ms ease-out;
}

@keyframes thinking {
    0%, 80%, 100% { opacity: 0.3; }
    40% { opacity: 1; }
}

@keyframes response-enter {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (min-width: 1101px) {
    [data-testid="stSidebar"] {
        width: 280px;
        min-width: 280px;
        max-width: 280px;
    }
}

@media (min-width: 768px) and (max-width: 1100px) {
    [data-testid="stSidebar"] {
        width: 260px;
        min-width: 260px;
        max-width: 260px;
    }

    [data-testid="stMainBlockContainer"] {
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
}

@media (max-width: 767px) {
    [data-testid="stMainBlockContainer"] {
        padding: 3.1rem 0.85rem 1rem;
    }

    .st-key-topbar {
        margin-bottom: 0.65rem;
    }

    .assistant-label {
        display: none;
    }

    .topbar-title .conversation-title {
        font-size: 13px;
    }

    .st-key-topbar [data-testid="stColumn"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }

    .st-key-topbar [data-testid="stColumn"]:last-child {
        flex: 0 0 56px !important;
    }

    .st-key-suggestions [data-testid="stHorizontalBlock"] {
        flex-direction: column;
        gap: 0.6rem;
    }

    .st-key-suggestions [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 0 !important;
    }

    .st-key-suggestions button {
        min-height: 68px;
    }

    .st-key-welcome {
        padding-top: 1.5rem;
    }

    .welcome-copy {
        margin-bottom: 1.25rem;
    }

    [class*="st-key-user_message_"] {
        max-width: 92%;
    }

    .st-key-conversation [data-testid="stChatMessage"] {
        gap: 0.5rem;
    }

    .st-key-conversation [data-testid="stMarkdownContainer"] p,
    .st-key-conversation [data-testid="stMarkdownContainer"] li {
        font-size: 15px;
    }

    [data-testid="stBottomBlockContainer"] {
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 22px;
    }

    [data-testid="stSidebar"] button {
        min-height: 44px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .stApp *,
    .stApp *::before,
    .stApp *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
}
</style>
"""

LOGO_SVG = """
<svg viewBox="0 0 40 40" width="100%" height="100%"
     fill="none" aria-hidden="true">
    <path d="M20 4L24 15L36 20L24 25L20 36L16 25L4 20L16 15Z"
          stroke="currentColor" stroke-width="2.2"
          stroke-linejoin="round"/>
    <circle cx="20" cy="20" r="4" fill="currentColor"/>
</svg>
"""


def inject_styles():
    st.markdown(CSS, unsafe_allow_html=True)

    palettes = {
        "Light": {
            "bg": "#ffffff",
            "sidebar": "#f7f7f8",
            "text": "#202123",
            "accent": "#16796f",
        },
        "Dark": {
            "bg": "#181a1b",
            "sidebar": "#121415",
            "text": "#eceeef",
            "accent": "#6ecdbb",
        },
    }

    palette = palettes.get(st.session_state.ui_theme)
    if palette:
        st.markdown(
            f"""
            <style>
            .stApp {{
                --background-color: {palette["bg"]};
                --secondary-background-color: {palette["sidebar"]};
                --text-color: {palette["text"]};
                --primary-color: {palette["accent"]};
                --bg: {palette["bg"]};
                --sidebar-bg: {palette["sidebar"]};
                --text: {palette["text"]};
                --accent: {palette["accent"]};
                --focus: {palette["accent"]};
                color-scheme: {
                    "dark" if st.session_state.ui_theme == "Dark"
                    else "light"
                };
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Helpers
# ============================================================

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def generate_title(user_message: str) -> str:
    title = user_message.strip()
    if len(title) > 40:
        title = title[:40].rstrip() + "..."
    return title or "New Chat"


def make_title(user_message: str) -> str:
    return generate_title(user_message)


def get_tool_label(tool_name: str) -> tuple[str, str]:
    return TOOL_DISPLAY.get(
        tool_name,
        ("🔧", f"Using tool: {tool_name}"),
    )


def get_message_role(message):
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    return None


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") in {"text", "output_text"}
            ):
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return "" if content is None else str(content)


def diagnostic_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def plain_button_label(value) -> str:
    text = " ".join(str(value).split())
    for character in "\\`*_{}[]()#+-.!|>":
        text = text.replace(character, "\\" + character)
    return text


def notify(message: str, level: str = "info"):
    st.session_state.notice = {
        "message": message,
        "level": level,
    }


def show_notice():
    notice = st.session_state.pop("notice", None)
    if not notice:
        return

    if notice["level"] == "error":
        st.error(notice["message"])
    else:
        icon = "✅" if notice["level"] == "success" else "ℹ️"
        st.toast(notice["message"], icon=icon)


def queue_prompt(prompt: str):
    st.session_state.pending_prompt = prompt


def export_transcript() -> str:
    title = st.session_state.get("active_title", "Conversation")
    lines = [f"# {title}", ""]

    for message in st.session_state.message_history:
        role = get_message_role(message)
        if role not in {"user", "assistant"}:
            continue

        text = content_to_text(message.content)
        if text.strip():
            lines.extend([
                f"## {'You' if role == 'user' else 'Assistant'}",
                "",
                text,
                "",
            ])

    return "\n".join(lines)


def reset_upload_selection():
    st.session_state.uploader_generation += 1
    st.session_state.pop("last_indexed_pdf", None)
    st.session_state.pop("last_indexed_stats", None)


# ============================================================
# Session state
# ============================================================

def initialize_session_state():
    defaults = {
        "pending_prompt": None,
        "indexed_uploads": {},
        "uploader_generation": 0,
        "show_tool_details": False,
        "notice": None,
        "ui_theme": "Streamlit theme",
        "conversation_search_query": "",   # renamed — avoids clash with widget key
        "active_title": "New Chat",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "thread_id" not in st.session_state:
        conversations = retrieve_all_threads()

        if conversations:
            conversation = conversations[0]
            st.session_state.thread_id = conversation["thread_id"]
            st.session_state.active_title = (
                conversation.get("title") or "New Chat"
            )
        else:
            thread_id = generate_thread_id()
            create_conversation(thread_id, "New Chat")
            st.session_state.thread_id = thread_id

    if "message_history" not in st.session_state:
        st.session_state.message_history = retrieve_thread_messages(
            st.session_state.thread_id
        )


# ============================================================
# Conversation management
# ============================================================

def create_new_chat():
    thread_id = generate_thread_id()

    try:
        create_conversation(thread_id, "New Chat")
    except Exception:
        LOGGER.exception("Could not create conversation")
        notify("Could not create a new chat. Please try again.", "error")
        return

    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    st.session_state.active_title = "New Chat"
    st.session_state.pending_prompt = None
    st.session_state.conversation_search_query = ""   # renamed
    reset_upload_selection()


def new_conversation():
    create_new_chat()


def load_conversation(thread_id: str):
    if thread_id == st.session_state.thread_id:
        return

    try:
        messages = retrieve_thread_messages(thread_id)
    except Exception:
        LOGGER.exception("Could not load conversation %s", thread_id)
        notify(
            "Something went wrong while loading this conversation. "
            "Please try again.",
            "error",
        )
        return

    st.session_state.thread_id = thread_id
    st.session_state.message_history = messages
    st.session_state.active_title = "Conversation"
    st.session_state.pending_prompt = None
    reset_upload_selection()


def get_conversations():
    try:
        conversations = retrieve_all_threads()
    except Exception:
        LOGGER.exception("Could not retrieve conversation list")
        return [], False

    for conversation in conversations:
        if conversation["thread_id"] == st.session_state.thread_id:
            st.session_state.active_title = (
                conversation.get("title") or "New Chat"
            )
            break

    return conversations, True


# ============================================================
# Documents
# ============================================================

def document_display_name(document, position: int) -> str:
    candidate = None

    if isinstance(document, (str, os.PathLike)):
        candidate = str(document)
    elif isinstance(document, dict):
        for field in (
            "original_filename",
            "filename",
            "file_name",
            "name",
            "source",
            "path",
            "pdf_path",
        ):
            if document.get(field):
                candidate = str(document[field])
                break

    if candidate:
        return candidate.replace("\\", "/").rsplit("/", 1)[-1]

    return f"Document {position}"


def render_file_chip(name: str, state: str = "Indexed"):
    safe_name = html.escape(name, quote=True)
    safe_state = html.escape(state, quote=True)

    st.markdown(
        f"""
        <div class="file-chip">
            <span aria-hidden="true">📄</span>
            <span class="file-name" title="{safe_name}">{safe_name}</span>
            <span class="file-state">{safe_state}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_documents():
    st.markdown("#### Documents")
    st.caption(
        "Upload and index a PDF, then ask about it in this conversation."
    )

    thread_id = st.session_state.thread_id
    uploader_key = (
        f"pdf_uploader_{thread_id}_"
        f"{st.session_state.uploader_generation}"
    )

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key=uploader_key,
        help="Documents are indexed for the active conversation.",
    )

    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        digest = hashlib.sha256(pdf_bytes).hexdigest()

        uploads = st.session_state.indexed_uploads.get(thread_id, {})
        already_indexed = digest in uploads

        if already_indexed:
            render_file_chip(uploaded_pdf.name)
        else:
            st.caption("Selected. Index the PDF to make it available in chat.")

        if st.button(
            "Index PDF",
            icon=":material/add:",
            key="index_pdf",
            disabled=already_indexed or not pdf_bytes,
            use_container_width=True,
        ):
            tmp_path = None

            try:
                with st.spinner("Indexing PDF..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as tmp:
                        tmp.write(pdf_bytes)
                        tmp_path = tmp.name

                    stats = build_rag_index(tmp_path, thread_id)

                thread_uploads = st.session_state.indexed_uploads.setdefault(
                    thread_id, {}
                )
                thread_uploads[digest] = {
                    "name": uploaded_pdf.name,
                    "source_name": Path(tmp_path).name,
                    "stats": stats,
                }

                st.session_state.last_indexed_pdf = uploaded_pdf.name
                st.session_state.last_indexed_stats = stats
                st.session_state.uploader_generation += 1

                notify(
                    f"{uploaded_pdf.name} was indexed successfully.",
                    "success",
                )

            except Exception:
                LOGGER.exception("PDF indexing failed for %s", thread_id)
                st.error(
                    "Could not index this PDF. Check that it is readable "
                    "and not password-protected, then try again."
                )
            else:
                st.rerun()
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        LOGGER.warning(
                            "Could not remove temporary PDF %s", tmp_path
                        )

    try:
        documents = list(list_rag_pdfs(thread_id))
    except Exception:
        LOGGER.exception("Could not list PDFs for %s", thread_id)
        st.error("Could not load indexed documents. Please try again.")
        return

    if not documents:
        st.caption("No indexed documents in this conversation.")
        return

    st.caption(f"{len(documents)} indexed document(s)")

    known_uploads = st.session_state.indexed_uploads.get(thread_id, {})
    name_map = {
        record["source_name"]: record["name"]
        for record in known_uploads.values()
        if isinstance(record, dict) and "source_name" in record
    }

    for position, document in enumerate(documents, start=1):
        name = document_display_name(document, position)
        render_file_chip(name_map.get(name, name))

    st.caption('Try: "Summarize the PDF" or "What does it say about...?"')

    with st.expander("Clear documents", expanded=False):
        st.caption(
            "Removes the document index for this conversation only. "
            "Chat messages are not deleted."
        )

        confirmed = st.checkbox(
            "I want to clear this conversation's documents",
            key=f"confirm_clear_{thread_id}_"
                f"{st.session_state.uploader_generation}",
        )

        if st.button(
            "Clear indexed documents",
            icon=":material/delete_outline:",
            key="clear_pdfs",
            disabled=not confirmed,
            use_container_width=True,
        ):
            try:
                removed = clear_rag_index(thread_id)
            except Exception:
                LOGGER.exception("Could not clear PDFs for %s", thread_id)
                st.error("Could not clear documents. Please try again.")
            else:
                st.session_state.indexed_uploads.pop(thread_id, None)
                reset_upload_selection()
                notify(f"Cleared {removed} PDF(s).", "success")
                st.rerun()


# ============================================================
# Capabilities, preferences, and export
# ============================================================

def render_capabilities():
    st.markdown("#### Capabilities")
    st.caption(
        "Ask naturally in chat. Tool execution and availability are "
        "handled by the existing backend."
    )

    for group_name, tool_names in TOOL_GROUPS.items():
        pills = []

        for name in tool_names:
            _, label = get_tool_label(name)
            readable_name = name.replace("_", " ")

            pills.append(
                '<span class="tool-pill" '
                f'title="{html.escape(label, quote=True)}">'
                f"{html.escape(readable_name)}</span>"
            )

        st.markdown(
            f'<div class="tool-group">{html.escape(group_name)}</div>'
            f'<div class="tool-pills">{"".join(pills)}</div>',
            unsafe_allow_html=True,
        )


def render_export(key: str):
    visible_messages = any(
        get_message_role(message) in {"user", "assistant"}
        and content_to_text(message.content).strip()
        for message in st.session_state.message_history
    )

    st.download_button(
        "Export conversation",
        data=export_transcript(),
        file_name=f"conversation-{st.session_state.thread_id}.md",
        mime="text/markdown",
        icon=":material/download:",
        key=key,
        disabled=not visible_messages,
        use_container_width=True,
    )


def render_preferences():
    st.markdown("#### Preferences")

    st.selectbox(
        "Appearance",
        options=["Streamlit theme", "Light", "Dark"],
        key="ui_theme",
        help="Use the app's theme or choose a local light/dark appearance.",
    )

    def update_tool_preference():
        st.session_state.show_tool_details = (
            st.session_state.tool_activity_widget
        )

    st.checkbox(
        "Show tool activity",
        value=st.session_state.show_tool_details,
        key="tool_activity_widget",
        on_change=update_tool_preference,
        help=(
            "Show expandable tool arguments and results. "
            "These can contain sensitive data."
        ),
    )

    st.caption(
        "Tool details are hidden by default. Temporary execution status "
        "is still shown while the assistant works."
    )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar(conversations, conversations_available: bool):
    with st.sidebar:
        with st.container(key="sidebar_shell"):
            st.markdown(
                f"""
                <div class="brand">
                    <span class="brand-mark">{LOGO_SVG}</span>
                    <span class="brand-name">{html.escape(APP_NAME)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(key="new_chat_action"):
                st.button(
                    "New chat",
                    icon=":material/edit_square:",
                    on_click=new_conversation,
                    key="new_chat",
                    use_container_width=True,
                )

            with st.container(key="conversation_search"):
                # No explicit key here — avoids clash with session state
                search = st.text_input(
                    "Search conversations",
                    placeholder="Search conversations",
                    label_visibility="collapsed",
                    icon=":material/search:",
                )

            st.markdown(
                '<div class="nav-label">Conversations</div>',
                unsafe_allow_html=True,
            )

            query = search.strip().casefold()
            filtered = [
                conversation
                for conversation in conversations
                if query in str(
                    conversation.get("title") or "New Chat"
                ).casefold()
            ]

            with st.container(key="conversation_list"):
                if not conversations_available:
                    st.caption("Could not load conversations.")
                    st.button(
                        "Retry",
                        key="retry_conversation_list",
                        use_container_width=True,
                    )
                elif not filtered:
                    st.caption(
                        "No matching conversations."
                        if query else "Your conversations will appear here."
                    )

                for conversation in filtered:
                    thread_id = conversation["thread_id"]
                    title = str(conversation.get("title") or "New Chat")
                    is_current = thread_id == st.session_state.thread_id

                    st.button(
                        plain_button_label(title),
                        icon=(
                            ":material/chat_bubble:"
                            if is_current
                            else ":material/chat_bubble_outline:"
                        ),
                        key=f"conv_{thread_id}",
                        type="primary" if is_current else "secondary",
                        help=f"Active conversation: {title}" if is_current else title,
                        on_click=load_conversation,
                        args=(thread_id,),
                        use_container_width=True,
                    )

            with st.container(key="sidebar_footer"):
                with st.popover(
                    "Documents",
                    icon=":material/attach_file:",
                    use_container_width=True,
                ):
                    render_documents()

                with st.popover(
                    "Capabilities",
                    icon=":material/grid_view:",
                    use_container_width=True,
                ):
                    render_capabilities()

                with st.popover(
                    "Settings & export",
                    icon=":material/settings:",
                    use_container_width=True,
                ):
                    render_preferences()
                    st.divider()
                    render_export("sidebar_export")

                st.markdown(
                    f'<div class="version-note">'
                    f'LangGraph assistant · Streamlit '
                    f'{html.escape(st.__version__)}</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# Header
# ============================================================

def render_header():
    title = html.escape(
        str(st.session_state.active_title),
        quote=True,
    )

    with st.container(key="topbar"):
        left, right = st.columns([12, 1], gap="small")

        with left:
            st.markdown(
                f"""
                <div class="topbar-title">
                    <span class="assistant-label">Assistant</span>
                    <span class="conversation-title" title="{title}">
                        {title}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            with st.popover(
                "",
                icon=":material/more_horiz:",
                help="Conversation actions",
                use_container_width=True,
            ):
                st.markdown("#### Conversation")
                render_export("header_export")

                st.button(
                    "New chat",
                    icon=":material/edit_square:",
                    on_click=new_conversation,
                    key="header_new_chat",
                    use_container_width=True,
                )

                st.caption(
                    "Documents, capabilities, and preferences "
                    "are available in the sidebar."
                )


# ============================================================
# Welcome screen
# ============================================================

def render_welcome():
    with st.container(key="welcome"):
        st.markdown(
            f"""
            <div class="welcome-copy">
                <div class="welcome-mark">{LOGO_SVG}</div>
                <h1>How can I help you?</h1>
                <p>
                    Ask a question, analyze a document, research a topic,
                    or work through a problem.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="suggestions"):
            for offset in range(0, len(SUGGESTIONS), 2):
                columns = st.columns(2, gap="small")

                for column, suggestion in zip(
                    columns, SUGGESTIONS[offset:offset + 2]
                ):
                    with column:
                        st.button(
                            f'**{suggestion["title"]}**  \n'
                            f'{suggestion["description"]}',
                            icon=f':material/{suggestion["icon"]}:',
                            key=f"suggestion_{offset}_{suggestion['icon']}",
                            on_click=queue_prompt,
                            args=(suggestion["prompt"],),
                            use_container_width=True,
                        )

        st.caption("PDFs can be uploaded using Documents in the sidebar.")


# ============================================================
# Tool activity and message rendering
# ============================================================

def render_tool_activity(tool_name: str, arguments=None, result=None):
    if not st.session_state.show_tool_details:
        return

    emoji, label = get_tool_label(tool_name)

    with st.expander(f"{emoji} {label} - Details", expanded=False):
        st.caption("Tool")
        st.code(tool_name or "tool", language="text")

        if arguments is not None:
            st.caption("Arguments")
            st.code(diagnostic_text(arguments), language="json")

        if result is not None:
            st.caption("Result")
            st.code(diagnostic_text(result), language="text")


def render_message(message, index):
    role = get_message_role(message)
    if role not in {"user", "assistant"}:
        return

    content = content_to_text(message.content)
    if not content.strip():
        return

    with st.container(key=f"{role}_message_{index}"):
        with st.chat_message(
            role,
            avatar=":material/auto_awesome:" if role == "assistant" else None,
        ):
            st.markdown(content)


def render_messages():
    history = st.session_state.message_history

    tool_results = {
        getattr(message, "tool_call_id", None): message
        for message in history
        if isinstance(message, ToolMessage)
        and getattr(message, "tool_call_id", None)
    }

    handled_result_ids = set()

    for index, message in enumerate(history):
        role = get_message_role(message)

        if role in {"user", "assistant"}:
            render_message(message, index)

        if role == "assistant" and st.session_state.show_tool_details:
            for call in getattr(message, "tool_calls", []) or []:
                call_id = call.get("id")
                result_message = tool_results.get(call_id)

                render_tool_activity(
                    call.get("name", "tool"),
                    arguments=call.get("args", {}),
                    result=(
                        result_message.content
                        if result_message is not None
                        else None
                    ),
                )

                if result_message is not None:
                    handled_result_ids.add(call_id)

        elif role == "tool" and st.session_state.show_tool_details:
            call_id = getattr(message, "tool_call_id", None)

            if call_id not in handled_result_ids:
                render_tool_activity(
                    getattr(message, "name", None) or "tool",
                    result=message.content,
                )


def show_activity(placeholder, label: str = "Thinking"):
    placeholder.markdown(
        f"""
        <div class="activity" role="status" aria-live="polite">
            <span>{html.escape(label)}</span>
            <span class="activity-dots" aria-hidden="true">
                <i></i><i></i><i></i>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Composer
# ============================================================

def render_composer():
    return st.chat_input(
        "Message the assistant",
        key=f"chat_composer_{st.session_state.thread_id}",
    )


# ============================================================
# Streaming
# ============================================================

def process_user_input(user_input: str):
    thread_id = st.session_state.thread_id
    is_first_message = not any(
        isinstance(message, HumanMessage)
        for message in st.session_state.message_history
    )

    user_message = HumanMessage(content=user_input)
    render_message(user_message, "live")
    st.session_state.message_history.append(user_message)

    if is_first_message:
        title = make_title(user_input)

        try:
            update_conversation_title(thread_id, title)
            st.session_state.active_title = title
        except Exception:
            LOGGER.exception("Could not update title for %s", thread_id)
            notify(
                "Your message was submitted, but the conversation title "
                "could not be updated."
            )

    stream_error = False
    refresh_error = False
    tool_status = None
    current_status = "Thinking"
    final_text_parts = []

    try:
        with st.container(key="assistant_message_live"):
            with st.chat_message(
                "assistant",
                avatar=":material/auto_awesome:",
            ):
                tool_status = st.empty()
                show_activity(tool_status)

                def response_stream():
                    nonlocal current_status

                    for chunk, meta in chatbot.stream(
                        {
                            "messages": [
                                HumanMessage(content=user_input)
                            ]
                        },
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                            }
                        },
                        stream_mode="messages",
                    ):
                        meta = meta or {}
                        node_name = meta.get("langgraph_node")

                        if isinstance(chunk, ToolMessage) or node_name == "tools":
                            current_status = "Thinking"
                            show_activity(tool_status, current_status)
                            continue

                        detected_tools = []

                        for field in ("tool_calls", "tool_call_chunks"):
                            for call in getattr(chunk, field, None) or []:
                                name = call.get("name")
                                if name and name not in detected_tools:
                                    detected_tools.append(name)

                        for name in detected_tools:
                            _, label = get_tool_label(name)
                            if label != current_status:
                                current_status = label
                                show_activity(tool_status, label)

                        chunk_type = getattr(chunk, "type", "")
                        if chunk_type in {
                            "human", "HumanMessageChunk",
                            "system", "SystemMessageChunk",
                            "tool", "ToolMessageChunk",
                        }:
                            continue

                        text = content_to_text(
                            getattr(chunk, "content", "")
                        )

                        if not text:
                            continue

                        if not detected_tools:
                            tool_status.empty()
                            current_status = ""

                        final_text_parts.append(text)
                        yield text

                st.write_stream(response_stream())

    except Exception:
        stream_error = True
        LOGGER.exception("Response generation failed for %s", thread_id)
        notify(
            "The response was interrupted. Messages saved by the backend "
            "will remain in this chat. You can ask the assistant to continue.",
            "error",
        )

    finally:
        if tool_status is not None:
            tool_status.empty()

    try:
        st.session_state.message_history = retrieve_thread_messages(thread_id)
    except Exception:
        refresh_error = True
        LOGGER.exception("Could not refresh history for %s", thread_id)
        notify(
            "The response finished, but conversation history could not "
            "be refreshed. Reload this conversation before sending again.",
            "error",
        )

    try:
        touch_conversation(thread_id)
    except Exception:
        LOGGER.exception("Could not update timestamp for %s", thread_id)
        if not stream_error and not refresh_error:
            notify(
                "The conversation was processed, but its recent activity "
                "could not be updated."
            )

    if refresh_error:
        show_notice()
        return

    st.rerun()


# ============================================================
# Application
# ============================================================

def main():
    try:
        initialize_session_state()
    except Exception:
        LOGGER.exception("Application initialization failed")
        st.error(
            "Could not open your conversations. Check the backend "
            "connection and try again."
        )
        if st.button("Try again", key="retry_initialization"):
            st.rerun()
        st.stop()

    inject_styles()

    conversations, conversations_available = get_conversations()
    render_sidebar(conversations, conversations_available)
    render_header()
    show_notice()

    queued_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    typed_prompt = render_composer()
    user_input = typed_prompt or queued_prompt

    visible_history = any(
        get_message_role(message) in {"user", "assistant"}
        and content_to_text(message.content).strip()
        for message in st.session_state.message_history
    )

    if not visible_history and not user_input:
        render_welcome()

    with st.container(key="conversation"):
        render_messages()

        if user_input:
            process_user_input(user_input)


if __name__ == "__main__":
    main()