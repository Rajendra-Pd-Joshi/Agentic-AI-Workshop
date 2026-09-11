import ast
import json
import io
import math
import os
import random
import re
import sqlite3
import subprocess
import tempfile
import uuid as uuid_module
from collections import Counter
from datetime import datetime
from typing import Annotated, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import hashlib
import threading

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAG pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


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
    Also create the memory table for persistent AI memory.
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

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()


initialize_database()


# ============================================================
# RAG Engine
# ============================================================
# Per-thread vector stores: { thread_id -> { pdf_hash -> FAISS } }
# This is in-memory so it resets on restart — PDF must be re-uploaded.
_rag_lock = threading.Lock()
_rag_stores: dict[str, dict[str, FAISS]] = {}

# Shared splitter and embeddings (lazy-init embeddings so no key error at import)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
)
_embeddings: OpenAIEmbeddings | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def build_rag_index(pdf_path: str, thread_id: str) -> dict:
    """
    Full RAG pipeline for one PDF:
      1. PyPDFLoader  — load document page by page
      2. RecursiveCharacterTextSplitter — chunk (1000 chars / 200 overlap)
      3. OpenAI embeddings  — embed every chunk
      4. FAISS vectorstore  — store & persist in memory (per thread)

    Returns a summary dict with stats.
    """
    # ── 1. Load ──────────────────────────────────────────────
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()                   # list[Document], one per page
    n_pages = len(pages)

    # ── 2. Chunk ─────────────────────────────────────────────
    chunks = _splitter.split_documents(pages)
    n_chunks = len(chunks)

    # ── 3 & 4. Embed + store ─────────────────────────────────
    vectorstore = FAISS.from_documents(chunks, _get_embeddings())

    # Compute a stable hash so we can tell if same PDF is re-uploaded
    with open(pdf_path, "rb") as fh:
        pdf_hash = hashlib.md5(fh.read()).hexdigest()

    with _rag_lock:
        if thread_id not in _rag_stores:
            _rag_stores[thread_id] = {}
        _rag_stores[thread_id][pdf_hash] = vectorstore

    return {
        "pdf_hash": pdf_hash,
        "n_pages": n_pages,
        "n_chunks": n_chunks,
        "thread_id": thread_id,
    }

import copy
def get_rag_retriever(thread_id: str, k: int = 7):
    """
    Return an MMR retriever that searches across ALL PDFs uploaded in this thread.
    Uses Maximal Marginal Relevance to pick top-k diverse, relevant chunks.
    Returns None if no PDFs are indexed for this thread.
    """
    with _rag_lock:
        stores = _rag_stores.get(thread_id, {})
        if not stores:
            return None
        store_list = list(stores.values())

    if len(store_list) == 1:
        vs = store_list[0]
    else:
        # FIX: Copy the first store so we don't mutate the cached version
        vs = copy.deepcopy(store_list[0])
        for extra in store_list[1:]:
            vs.merge_from(extra)

    # MMR retriever: fetch_k candidates, return k diverse results
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,             # final docs returned  (5-10 per spec)
            "fetch_k": k * 4,   # candidate pool for MMR diversity calc
            "lambda_mult": 0.6, # 0=max diversity, 1=max relevance
        },
    )


def list_rag_pdfs(thread_id: str) -> list[str]:
    """Return the md5 hashes of all PDFs indexed for this thread."""
    with _rag_lock:
        return list(_rag_stores.get(thread_id, {}).keys())


def clear_rag_index(thread_id: str, pdf_hash: str = "") -> int:
    """
    Remove one PDF (by hash) or all PDFs for a thread.
    Returns number of stores removed.
    """
    with _rag_lock:
        stores = _rag_stores.get(thread_id, {})
        if pdf_hash:
            if pdf_hash in stores:
                del stores[pdf_hash]
                return 1
            return 0
        removed = len(stores)
        _rag_stores[thread_id] = {}
        return removed


# ============================================================
# LangGraph State
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# ── ORIGINAL TOOLS ──────────────────────────────────────────
# ============================================================

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

@tool
def get_current_datetime(tz_name: str = "UTC") -> str:
    """
    Get the current date and time for a given timezone.
    Examples of valid timezones: 'UTC', 'US/Eastern', 'Asia/Tokyo', 'Europe/London', 'Asia/Kathmandu'.
    Defaults to UTC if no timezone is provided.
    """
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        human_format = now.strftime(f"%A, %B %d, %Y — %I:%M:%S %p ({tz_name})")
        iso_format = now.isoformat()
        return f"{human_format} | ISO: {iso_format}"
    except ZoneInfoNotFoundError:
        # FIX: Replaced deprecated datetime.utcnow() with datetime.now(timezone.utc)
        now = datetime.now(timezone.utc) 
        human_format = now.strftime('%A, %B %d, %Y — %I:%M:%S %p UTC')
        iso_format = now.isoformat()
        return (
            f"Timezone '{tz_name}' not recognized. "
            f"Falling back to UTC: {human_format} | ISO: {iso_format}"
        )


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, //, %, **, abs, round, min, max,
              sqrt, sin, cos, tan, log, log10, log2, exp,
              ceil, floor, pi, e, inf.
    Example inputs: '2 ** 10', 'sqrt(144) + log(100, 10)', 'round(3.14159, 2)'
    """
    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "asin": math.asin, "acos": math.acos,
        "atan": math.atan, "log": math.log, "log10": math.log10,
        "log2": math.log2, "exp": math.exp, "pow": math.pow,
        "ceil": math.ceil, "floor": math.floor, "factorial": math.factorial,
        "pi": math.pi, "e": math.e, "inf": math.inf, "tau": math.tau,
    }
    try:
        # FIX 1: Remove 'math.' prefix so the AST attribute blocker isn't triggered
        clean_expression = expression.strip().replace("math.", "")
        
        # FIX 2: Replace '^' with '**' because LLMs often use '^' for exponentiation
        clean_expression = clean_expression.replace("^", "**")
        
        tree = ast.parse(clean_expression, mode="eval")
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Error: Import statements are not allowed."
            if isinstance(node, ast.Attribute):
                return "Error: Attribute access is not allowed."
                
        result = eval(compile(tree, "<expr>", "eval"), safe_globals)
        return f"{expression} = {result}"
        
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather conditions for any city in the world.
    Uses the Open-Meteo API — no API key required.
    Example: city='Kathmandu' or city='Tokyo'
    """
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            return f"City '{city}' could not be found."
            
        loc = geo_data["results"][0]
        # FIX 1: Safely get lat/lon with .get() instead of direct key access
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            return f"Coordinates for '{city}' are missing from the geocoding service."
            
        display_name = loc.get("name", city)
        country = loc.get("country", "")
        
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,cloud_cover",
                "temperature_unit": "celsius", "wind_speed_unit": "kmh",
            },
            timeout=10,
        )
        w_data = weather_resp.json()
        
        # FIX 2: Safely check both 'current' and 'current_weather' schemas
        w = w_data.get("current") or w_data.get("current_weather", {})
        if not w:
            return f"Weather data format unrecognized for '{city}'."
            
        code = w.get("weather_code", w.get("weathercode", -1))
        
        condition_map = {
            0: "Clear sky ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
            45: "Foggy 🌫️", 48: "Icy fog 🌫️",
            51: "Light drizzle 🌦️", 53: "Moderate drizzle 🌦️", 55: "Dense drizzle 🌧️",
            61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
            71: "Slight snow 🌨️", 73: "Moderate snow 🌨️", 75: "Heavy snow ❄️",
            80: "Slight showers 🌦️", 81: "Moderate showers 🌧️", 82: "Violent showers ⛈️",
            95: "Thunderstorm ⛈️", 96: "Thunderstorm w/ hail ⛈️", 99: "Heavy thunderstorm ⛈️",
        }
        condition = condition_map.get(code, f"Code {code}")
        
        return (
            f"🌍 Weather in {display_name}, {country}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Condition:     {condition}\n"
            f"• Temperature:   {w.get('temperature_2m', 'N/A')}°C (feels like {w.get('apparent_temperature', 'N/A')}°C)\n"
            f"• Humidity:      {w.get('relative_humidity_2m', 'N/A')}%\n"
            f"• Wind Speed:    {w.get('wind_speed_10m', 'N/A')} km/h\n"
            f"• Cloud Cover:   {w.get('cloud_cover', 'N/A')}%\n"
            f"• Precipitation: {w.get('precipitation', 'N/A')} mm"
        )
    except requests.Timeout:
        return "Weather request timed out. Please try again."
    except Exception as exc:
        return f"Weather fetch error: {exc}"


@tool
def search_wikipedia(query: str, sentences: int = 4) -> str:
    """
    Search Wikipedia and return a concise summary of the topic.
    Args:
        query: The topic or question to search for.
        sentences: Number of summary sentences to return (default 4).
    """
    try:
        import wikipedia
        import re
        
        # FIX 1: Set a custom user agent to prevent Wikipedia from blocking the request (403 Forbidden)
        wikipedia.set_user_agent("AgenticAIChatbot/1.0 (admin@localhost)")
        
        results = wikipedia.search(query, results=5)
        if not results:
            return f"No Wikipedia articles found for '{query}'."
            
        for candidate in results:
            try:
                page = wikipedia.page(candidate, auto_suggest=False)
                
                # FIX 2: Do not call wikipedia.summary() which triggers a second network request.
                # Instead, parse the summary directly from the already-fetched page object.
                raw_summary = page.summary
                
                # Simple sentence splitter based on periods (not perfect, but fast)
                sentence_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_summary) if s.strip()]
                short_summary = " ".join(sentence_list[:sentences])
                
                return f"📖 **{page.title}**\n\n{short_summary}\n\n🔗 Source: {page.url}"
                
            except wikipedia.exceptions.DisambiguationError:
                # If a page is ambiguous, skip to the next search result
                continue
            except wikipedia.exceptions.PageError:
                # If the page doesn't exist, skip to the next search result
                continue
                
        return f"Could not load a Wikipedia article for '{query}'."
        
    except ImportError:
        return "The 'wikipedia' package is not installed. Run: pip install wikipedia"
    except Exception as exc:
        return f"Wikipedia error: {exc}"

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information.
    Automatically uses Tavily if TAVILY_API_KEY is in the environment,
    otherwise falls back to a rate-limit-resistant DuckDuckGo search.
    Args:
        query: The search query.
        max_results: Number of results to return (default 5, max 10).
    """
    max_results = min(max_results, 10)
    
    # --- Strategy 1: Tavily (If API Key is available) ---
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            # or simply use the requests library to hit tavily directly to avoid new dependencies
            # Actually using requests is safer to avoid forcing them to pip install another langchain package if they don't want to.
            # Wait, using requests for Tavily is super easy:
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": tavily_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "max_results": max_results
            }
            resp = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for r in data.get("results", []):
                    results.append(f"📌 **{r.get('title')}**\n   {r.get('content')}\n   🔗 {r.get('url')}")
                if results:
                    return f"🔍 Web results for: *{query}* (via Tavily)\n\n" + "\n\n".join(results)
        except Exception as e:
            pass # Fall back to DDG if Tavily fails

    # --- Strategy 2: DuckDuckGo 'lite' Backend (No API Key needed) ---
    try:
        from duckduckgo_search import DDGS
        import time
        
        results = []
        with DDGS() as ddgs:
            # Using 'lite' backend bypasses the heavy JS bot-checks
            for r in ddgs.text(query, max_results=max_results, backend="lite"):
                results.append(f"📌 **{r['title']}**\n   {r['body']}\n   🔗 {r['href']}")
        
        if results:
            return f"🔍 Web search results for: *{query}*\n\n" + "\n\n".join(results)
    except Exception as e:
        pass

    # --- Strategy 3: DuckDuckGo News (If text is rate-limited) ---
    try:
        from duckduckgo_search import DDGS
        import time
        time.sleep(1) # Small delay to prevent immediate secondary rate limit
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                date = r.get("date", "")[:10]
                results.append(
                    f"📰 **{r['title']}** ({date})\n"
                    f"   {r.get('body', r.get('excerpt', ''))}\n"
                    f"   Source: {r.get('source', 'Unknown')} · 🔗 {r['url']}"
                )
        if results:
            return f"🔍 News search results for: *{query}*\n\n" + "\n\n".join(results)
    except Exception as e:
        pass

    return (
        f"⚠️ All search providers failed for '{query}'.\n"
        f"If you are getting rate-limited constantly, consider getting a free API key "
        f"from tavily.com and adding TAVILY_API_KEY to your .env file."
    )

import email.utils
import re
import requests

@tool
def get_news(topic: str = "", country: str = "", max_results: int = 8) -> str:
    """
    Fetch the latest news headlines using RSS feeds from major news sources.
    Works without any API key. Great for current events, country-specific news,
    or any topic like 'Nepal', 'technology', 'sports', 'business', etc.
    Args:
        topic: News topic or keyword (e.g. 'Nepal', 'earthquake', 'politics').
               Leave blank for top world headlines.
        country: Country-specific news (e.g. 'Nepal', 'India', 'US').
                 Leave blank for international sources.
        max_results: Number of headlines to return (default 8, max 20).
    """
    try:
        import feedparser
    except ImportError:
        return "The 'feedparser' package is not installed. Run: pip install feedparser"

    max_results = max(1, min(max_results, 20))
    combined = f"{topic} {country}".lower().strip()

    # Regional Feeds
    COUNTRY_FEEDS = {
        "nepal": [
            ("The Himalayan Times", "https://thehimalayantimes.com/feed"),
            ("Kathmandu Post", "https://kathmandupost.com/rss"),
            ("OnlineKhabar English", "https://english.onlinekhabar.com/feed"),
            ("Setopati English", "https://setopati.com/feed"),
        ],
        "india": [
            ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
            ("The Hindu", "https://www.thehindu.com/feeder/default.rss"),
            ("NDTV", "https://feeds.feedburner.com/ndtvnews-top-stories"),
        ],
        "us": [
            ("NPR", "https://feeds.npr.org/1001/rss.xml"),
            ("CNN Top Stories", "http://rss.cnn.com/rss/edition.rss"),
        ],
        "uk": [
            ("BBC News", "https://feeds.bbci.co.uk/news/rss.xml"),
            ("The Guardian World", "https://www.theguardian.com/world/rss"),
        ],
    }

    # Universal / Search-driven feeds
    query_param = topic or country or "world news"
    GENERAL_FEEDS = [
        ("Google News", f"https://news.google.com/rss/search?q={requests.utils.quote(query_param)}&hl=en&gl=US&ceid=US:en"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ]

    # Select target feeds
    feeds_to_try = []
    for key, feed_list in COUNTRY_FEEDS.items():
        if key in combined:
            feeds_to_try = feed_list + GENERAL_FEEDS
            break
    if not feeds_to_try:
        feeds_to_try = GENERAL_FEEDS

    # Realistic browser headers to prevent 403 Forbidden errors
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    all_entries = []
    sources_used = []

    for feed_name, feed_url in feeds_to_try:
        # Stop fetching once we have gathered more than enough candidates
        if len(all_entries) >= max_results * 2:
            break

        try:
            # Low timeout (4s) so stalled feeds do not hang the whole tool
            resp = requests.get(feed_url, headers=headers, timeout=4)
            if resp.status_code != 200:
                continue

            feed = feedparser.parse(resp.content)
            if not feed.entries:
                continue

            sources_used.append(feed_name)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                summary = entry.get("summary", entry.get("description", "")).strip()
                summary = re.sub(r"<[^>]+>", "", summary)[:200]

                pub = entry.get("published", entry.get("updated", ""))
                try:
                    pub_dt = email.utils.parsedate_to_datetime(pub)
                    pub_str = pub_dt.strftime("%b %d, %Y %H:%M")
                except Exception:
                    pub_str = pub[:16] if pub else ""

                if topic and feed_name != "Google News":
                    haystack = (title + " " + summary).lower()
                    if not any(kw in haystack for kw in topic.lower().split()):
                        continue

                all_entries.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "pub": pub_str,
                    "source": feed_name,
                })
        except Exception:
            continue

    # Fallback to DuckDuckGo news if RSS feeds were blocked or returned nothing
    if not all_entries:
        try:
            from duckduckgo_search import DDGS
            search_q = f"{topic} {country} news".strip()
            with DDGS() as ddgs:
                for r in ddgs.news(search_q, max_results=max_results):
                    date = r.get("date", "")[:10]
                    all_entries.append({
                        "title": r.get("title", ""),
                        "link": r.get("url", ""),
                        "summary": r.get("body", "")[:200],
                        "pub": date,
                        "source": r.get("source", "DuckDuckGo News"),
                    })
            if all_entries:
                sources_used.append("DuckDuckGo News")
        except Exception:
            pass

    if not all_entries:
        return (
            f"⚠️ Could not fetch news for '{topic or country or 'world'}' right now.\n"
            f"Providers timed out or rate-limited. Try using `fetch_webpage` with a direct URL."
        )

    # Deduplicate entries by title
    seen = set()
    unique_entries = []
    for e in all_entries:
        clean_title = re.sub(r"[^\w\s]", "", e["title"]).lower()[:50]
        if clean_title not in seen:
            seen.add(clean_title)
            unique_entries.append(e)

    top = unique_entries[:max_results]
    lines = []
    for e in top:
        pub_str = f" · {e['pub']}" if e["pub"] else ""
        summary_str = f"\n   {e['summary']}" if e["summary"] else ""
        lines.append(f"📰 **{e['title']}**{summary_str}\n   {e['source']}{pub_str} · 🔗 {e['link']}")

    label = " | ".join(dict.fromkeys(sources_used))
    header_topic = f"{topic} {country}".strip() or "World"
    return (
        f"📰 **Latest News — {header_topic}** ({len(top)} headlines)\n"
        f"Sources: {label}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + "\n\n".join(lines)
    )

import requests

@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and extract readable text content from any public webpage URL.
    Useful for reading articles, documentation, or links shared by the user.
    Args:
        url: The full URL including https:// or http://
    """
    try:
        from bs4 import BeautifulSoup

        # Modern browser request headers to avoid 403 Forbidden / bot blocks
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }

        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        # Fix encoding issues (prevents garbled apostrophes/quotes)
        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip non-content and clutter elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "svg", "noscript"]):
            tag.decompose()

        # Prioritize core content containers
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(id="main-content")
            or soup.find(class_="content")
            or soup.find(class_="post-content")
            or soup.find(class_="article-body")
            or soup.body
        )

        text = (main or soup).get_text(separator="\n", strip=True)

        # Remove excessive whitespace while preserving readable paragraphs
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        if not cleaned_text:
            return f"⚠️ Could not extract readable text from {url}. The site may require JavaScript rendering."

        if len(cleaned_text) > 4000:
            cleaned_text = cleaned_text[:4000] + "\n\n… [Content truncated to 4000 characters]"

        return f"📄 Content from {url}:\n\n{cleaned_text}"

    except requests.HTTPError as exc:
        return f"HTTP error fetching page ({exc.response.status_code}): {exc}"
    except requests.Timeout:
        return "Request timed out. The website server may be slow or blocking automated traffic."
    except ImportError:
        return "The 'beautifulsoup4' package is not installed. Run: pip install beautifulsoup4"
    except Exception as exc:
        return f"Webpage fetch error: {exc}"


import re

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a numeric value between common units of measurement.
    Supported categories:
      - Length: km, miles, meters, feet, inches, cm, mm, yards
      - Weight/Mass: kg, lbs, grams, oz, mg, tonnes
      - Volume: liters, ml, gallons, cups, pints, quarts, fl_oz
      - Temperature: celsius, fahrenheit, kelvin
      - Speed: kmh, mph, ms (meters per second), knots
      - Area: sqm, sqft, sqkm, sqmiles, acres, hectares
      - Time: seconds, minutes, hours, days, weeks, years
      - Data: bytes, kb, mb, gb, tb
    """
    
    # 1. Standardize formatting and strip punctuation (e.g. "lbs." -> "lbs", "sq ft" -> "sqft")
    f = re.sub(r'[^a-z0-9]', '', str(from_unit).lower().strip())
    t = re.sub(r'[^a-z0-9]', '', str(to_unit).lower().strip())

    # 2. Map aliases to standard internal keys
    aliases = {
        # Temperature
        "c": "celsius", "f": "fahrenheit", "k": "kelvin",
        # Length
        "m": "meters", "meter": "meters", "km": "kilometers", "kms": "kilometers", 
        "mile": "miles", "ft": "feet", "foot": "feet", "in": "inches", "inch": "inches",
        "cm": "centimeters", "mm": "millimeters", "yd": "yards", "yard": "yards",
        # Weight
        "g": "grams", "gram": "grams", "kg": "kilograms", "kgs": "kilograms",
        "lb": "pounds", "lbs": "pounds", "oz": "ounces", "ounce": "ounces", "mg": "milligrams",
        # Volume
        "l": "liters", "liter": "liters", "ml": "milliliters", "gal": "gallons", "gallon": "gallons",
        "floz": "floz", "cup": "cups", "pint": "pints", "quart": "quarts",
        # Speed
        "ms": "ms", "meterspersecond": "ms", "kmh": "kmh", "mph": "mph", "knot": "knots",
        # Area
        "m2": "sqm", "sqmeter": "sqm", "sqft": "sqft", "sqfoot": "sqft", "sqkm": "sqkm", 
        "sqmile": "sqmiles", "sqmiles": "sqmiles", "acre": "acres", "ha": "hectares",
        # Time
        "s": "seconds", "sec": "seconds", "secs": "seconds", "min": "minutes", "mins": "minutes",
        "hr": "hours", "hrs": "hours", "hour": "hours", "day": "days", "wk": "weeks", "yr": "years",
        # Data
        "b": "bytes", "byte": "bytes"
    }

    f_std = aliases.get(f, f)
    t_std = aliases.get(t, t)

    # 3. Temperature Conversions (Special formulas)
    temp_conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"):     lambda v: v + 273.15,
        ("kelvin", "celsius"):     lambda v: v - 273.15,
        ("fahrenheit", "kelvin"):  lambda v: (v - 32) * 5 / 9 + 273.15,
        ("kelvin", "fahrenheit"):  lambda v: (v - 273.15) * 9 / 5 + 32,
    }
    
    if (f_std, t_std) in temp_conversions:
        result = temp_conversions[(f_std, t_std)](value)
        unit_symbols = {"celsius": "°C", "fahrenheit": "°F", "kelvin": "K"}
        fs, ts = unit_symbols.get(f_std, f_std), unit_symbols.get(t_std, t_std)
        # Using %g removes trailing zeros for a clean output (e.g. 10.0 becomes 10)
        return f"{value:g}{fs} = {result:g}{ts}"

    if f_std == t_std:
        return f"{value:g} {from_unit} = {value:g} {to_unit}"

    # 4. Multiplier Tables (Base unit to target conversion)
    length = {"meters": 1, "kilometers": 1000, "miles": 1609.344, "feet": 0.3048, "inches": 0.0254, "centimeters": 0.01, "millimeters": 0.001, "yards": 0.9144, "nauticalmiles": 1852}
    weight = {"grams": 1, "kilograms": 1000, "pounds": 453.59237, "ounces": 28.3495, "milligrams": 0.001, "tonnes": 1000000}
    volume = {"milliliters": 1, "liters": 1000, "gallons": 3785.41, "cups": 236.588, "pints": 473.176, "quarts": 946.353, "floz": 29.5735, "tbsp": 14.7868, "tsp": 4.92892}
    speed  = {"ms": 1, "kmh": 0.277778, "mph": 0.44704, "knots": 0.514444, "fps": 0.3048}
    area   = {"sqm": 1, "sqft": 0.092903, "sqkm": 1000000, "sqmiles": 2589988.11, "acres": 4046.86, "hectares": 10000}
    time_t = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800, "years": 31536000}
    data   = {"bytes": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}

    # 5. Execute Conversion
    for category_name, table in [("Length", length), ("Weight", weight), ("Volume", volume), ("Speed", speed), ("Area", area), ("Time", time_t), ("Data", data)]:
        if f_std in table and t_std in table:
            # Formula: (Value * Source_Base_Multiplier) / Target_Base_Multiplier
            result = value * table[f_std] / table[t_std]
            return f"{value:g} {from_unit} = {result:g} {to_unit}"

    return (
        f"Conversion from '{from_unit}' to '{to_unit}' is not supported or crosses categories.\n"
        f"Supported Categories: Length, Weight, Volume, Temperature, Speed, Area, Time, Data."
    )

import re
from collections import Counter

@tool
def analyze_text(text: str) -> str:
    """
    Analyze a block of text and return detailed statistics.
    Returns: character count, word count, unique words, sentence count,
             paragraph count, average word length, most common words, and estimated reading time.
    """
    if not text or not text.strip():
        return "⚠️ Error: The provided text is empty."

    # 1. Better word extraction
    words_raw = text.split()
    words_clean = [re.sub(r"[^\w']", "", w).lower() for w in words_raw]
    words_clean = [w for w in words_clean if w]
    
    # 2. Better sentence splitting (handles decimals and ellipses better)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    unique_words = set(words_clean)
    avg_word_len = round(sum(len(w) for w in words_clean) / len(words_clean), 2) if words_clean else 0
    
    # 3. Estimated Reading Time (Average adult reads ~238 WPM)
    reading_time_minutes = max(1, round(len(words_raw) / 238))
    
    # 4. Stopwords filtering
    stopwords = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","was","are","were","it","its","this","that","i","you","he","she","we","they","be","been","have","has","had","do","does","did","will","would","could","should","may","might","not","from","by","as","so","if","my","your","our","their","his","her"}
    content_words = [w for w in words_clean if w not in stopwords and len(w) > 2]
    
    top_words = Counter(content_words).most_common(5)
    top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top_words) if top_words else "N/A"
    
    chars_no_spaces = len(re.sub(r'\s+', '', text))

    return (
        f"📊 **Text Analysis**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Total characters:        {len(text)}\n"
        f"• Characters (no spaces):  {chars_no_spaces}\n"
        f"• Total words:             {len(words_raw)}\n"
        f"• Unique words:            {len(unique_words)}\n"
        f"• Sentences:               {len(sentences)}\n"
        f"• Paragraphs:              {len(paragraphs)}\n"
        f"• Avg word length:         {avg_word_len} chars\n"
        f"• Avg words/sentence:      {round(len(words_raw) / max(len(sentences), 1), 1)}\n"
        f"• Est. Reading Time:       ~{reading_time_minutes} min\n"
        f"• Top content words:       {top_str}"
    )

@tool
def generate_uuid(count: int = 1) -> str:
    """
    Generate one or more random UUIDs (universally unique identifiers).
    Useful for creating unique IDs for database records, sessions, or tokens.
    Args:
        count: Number of UUIDs to generate (default 1, max 10).
    """
    count = max(1, min(count, 10))
    ids = [str(uuid_module.uuid4()) for _ in range(count)]
    if count == 1:
        return f"Generated UUID:\n{ids[0]}"
    return "Generated UUIDs:\n" + "\n".join(f"{i + 1}. {uid}" for i, uid in enumerate(ids))


@tool
def random_number(minimum: float = 0, maximum: float = 100, count: int = 1, as_integer: bool = True) -> str:
    """
    Generate one or more random numbers within a given range.
    Args:
        minimum: Lower bound (default 0).
        maximum: Upper bound (default 100).
        count: How many numbers to generate (default 1, max 20).
        as_integer: If True (default), returns whole numbers; otherwise returns floats.
    """
    count = max(1, min(count, 20))
    if minimum > maximum:
        return f"Error: minimum ({minimum}) must be less than maximum ({maximum})."
    results = []
    for _ in range(count):
        if as_integer:
            results.append(str(random.randint(int(minimum), int(maximum))))
        else:
            results.append(f"{round(random.uniform(minimum, maximum), 6)}")
    kind = "integer" if as_integer else "float"
    label = f"random {kind}" + ("s" if count > 1 else "")
    return f"🎲 Generated {count} {label} in range [{minimum}, {maximum}]:\n" + ", ".join(results)


# ============================================================
# ── NEW: KNOWLEDGE & RESEARCH TOOLS ─────────────────────────
# ============================================================
import re

@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers on any topic (AI, physics, math, CS, etc.).
    Returns paper titles, authors, abstracts, and links.
    Args:
        query: Research topic or keywords.
        max_results: Number of papers to return (default 5, max 10).
    """
    try:
        import arxiv
        max_results = max(1, min(max_results, 10))
        
        # FIX 1: Use a custom client with retries to handle arXiv's unstable API
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3.0,
            num_retries=3
        )
        
        search = arxiv.Search(
            query=query, 
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
                
            # FIX 2: Strip hard-coded newlines from the abstract so it renders cleanly
            raw_abstract = paper.summary.replace("\n", " ")
            abstract = re.sub(r'\s+', ' ', raw_abstract).strip()
            
            # Truncate slightly to save tokens in the LLM's context window
            if len(abstract) > 400:
                abstract = abstract[:397] + "..."
                
            results.append(
                f"📄 **{paper.title}**\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   Abstract: {abstract}\n"
                f"   🔗 {paper.entry_id}"
            )
            
        if not results:
            return f"No arXiv papers found for '{query}'."
            
        return f"📚 **arXiv results for: *{query}***\n\n" + "\n\n".join(results)
        
    except ImportError:
        return "The 'arxiv' package is not installed. Run: pip install arxiv"
    except Exception as exc:
        return f"arXiv search error: {exc}"

import os
import requests

@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for medical and biomedical research papers via NCBI Entrez API.
    Args:
        query: Medical or biological research topic.
        max_results: Number of results to return (default 5, max 10).
    """
    try:
        max_results = max(1, min(max_results, 10))

        # NCBI Entrez parameters to prevent 429 rate limits & IP bans
        email = os.getenv("NCBI_EMAIL", "agent@localhost")
        tool_name = "AgenticAIChatbot"
        api_key = os.getenv("NCBI_API_KEY", "")

        base_params = {
            "db": "pubmed",
            "retmode": "json",
            "tool": tool_name,
            "email": email,
        }
        if api_key:
            base_params["api_key"] = api_key

        # Step 1: Search for PubMed IDs (PMIDs)
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            **base_params,
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
        }

        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed articles found for '{query}'."

        # Step 2: Fetch summaries for returned IDs
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            **base_params,
            "id": ",".join(ids),
        }

        summary_resp = requests.get(summary_url, params=summary_params, timeout=10)
        summary_resp.raise_for_status()
        summary_data = summary_resp.json().get("result", {})

        results = []
        for uid in ids:
            article = summary_data.get(uid, {})
            title = article.get("title", "Unknown title").strip().rstrip(".")
            authors = article.get("authors", [])
            author_names = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_names += " et al."
            if not author_names:
                author_names = "Unknown authors"

            pub_date = article.get("pubdate", "Unknown date")
            source = article.get("source", "NCBI")

            results.append(
                f"🔬 **{title}**\n"
                f"   Authors: {author_names}\n"
                f"   Published: {pub_date} ({source})\n"
                f"   🔗 https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            )

        return f"🏥 **PubMed results for: *{query}***\n\n" + "\n\n".join(results)

    except requests.HTTPError as exc:
        return f"PubMed API HTTP error ({exc.response.status_code}): {exc}"
    except requests.Timeout:
        return "PubMed API request timed out. NCBI servers may be busy."
    except Exception as exc:
        return f"PubMed search error: {exc}"


import html
import requests

@tool
def stackoverflow_search(query: str, max_results: int = 5) -> str:
    """
    Search Stack Overflow for programming questions and answers.
    Args:
        query: Programming question or error message to search.
        max_results: Number of results to return (default 5, max 10).
    """
    try:
        max_results = max(1, min(max_results, 10))
        url = "https://api.stackexchange.com/2.3/search/advanced"
        
        # FIX 3: Removed 'filter="withbody"' to drastically reduce latency and payload size
        resp = requests.get(url, params={
            "order": "desc", 
            "sort": "relevance", 
            "q": query,
            "site": "stackoverflow", 
            "pagesize": max_results
        }, timeout=10)
        
        resp.raise_for_status()
        data = resp.json()
        
        # FIX 2: Explicitly catch Stack Exchange API rate limits and errors
        if "error_id" in data:
            return f"Stack Overflow API Error {data['error_id']}: {data.get('error_message')}"
            
        items = data.get("items", [])
        if not items:
            return f"No Stack Overflow results found for '{query}'."
            
        results = []
        for item in items:
            # FIX 1: Unescape HTML entities (e.g., &#39; to ')
            raw_title = item.get("title", "Unknown")
            title = html.unescape(raw_title)
            
            link = item.get("link", "")
            score = item.get("score", 0)
            answered = "✅ Answered" if item.get("is_answered") else "❓ Unanswered"
            answer_count = item.get("answer_count", 0)
            
            results.append(
                f"💻 **{title}**\n"
                f"   {answered} · {answer_count} answers · Score: {score}\n"
                f"   🔗 {link}"
            )
            
        return f"🔍 **Stack Overflow results for: *{query}***\n\n" + "\n\n".join(results)
        
    except requests.HTTPError as exc:
        return f"Stack Overflow API HTTP error ({exc.response.status_code}): {exc}"
    except requests.Timeout:
        return "Stack Overflow API request timed out."
    except Exception as exc:
        return f"Stack Overflow search error: {exc}"


# ============================================================
# ── NEW: CODE & DEVELOPER TOOLS ─────────────────────────────
# ============================================================

import os
import sys
import subprocess
import tempfile

@tool
def run_python(code: str) -> str:
    """
    Execute Python code in a sandboxed subprocess and return the output.
    Use this for data processing, calculations, generating outputs, etc.
    Args:
        code: Valid Python code to execute. Use print() to show results.
    Warning: Code runs in an isolated subprocess with a 10-second timeout.
    """
    tmp_path = None
    try:
        # FIX 2: Explicitly set encoding to utf-8 so emojis/symbols don't crash Windows terminals
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
            
        # FIX 1: Use sys.executable instead of "python3" so it works seamlessly on Windows (.venv)
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, 
            text=True, 
            encoding="utf-8",
            errors="replace",
            timeout=10
        )
        
        output = result.stdout.strip()
        errors = result.stderr.strip()
        
        if errors and not output:
            return f"❌ Error:\n```\n{errors}\n```"
        if errors:
            return f"⚠️ Output:\n```\n{output}\n```\n\nStderr:\n```\n{errors}\n```"
        if not output:
            return "✅ Code executed successfully (no output produced)."
        return f"✅ Output:\n```\n{output}\n```"
        
    except subprocess.TimeoutExpired:
        return "❌ Code execution timed out after 10 seconds."
    except Exception as exc:
        return f"Code execution error: {exc}"
    finally:
        # FIX 3: Guarantee temp file cleanup even if timeouts/errors happen
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

@tool
def github_search(
    query: str,
    search_type: str = "repositories",
    max_results: int = 5
) -> str:
    """
    Search GitHub for repositories, code, issues, or users.

    Args:
        query: GitHub search query, e.g. "langgraph python".
        search_type: One of:
            - repositories
            - code
            - issues
            - users
        max_results: Number of results to return (1-10).

    Returns:
        Formatted GitHub search results.
    """

    try:
        # ========================================================
        # 1. Normalize search type
        # ========================================================

        search_type = str(search_type).strip().lower()

        aliases = {
            "repo": "repositories",
            "repos": "repositories",
            "repository": "repositories",
            "repositories": "repositories",

            "code": "code",

            "issue": "issues",
            "issues": "issues",

            "user": "users",
            "users": "users",
        }

        if search_type not in aliases:
            return (
                "❌ Invalid GitHub search type.\n\n"
                f"You provided: `{search_type}`\n\n"
                "Valid options are:\n"
                "- `repositories`\n"
                "- `code`\n"
                "- `issues`\n"
                "- `users`"
            )

        search_type = aliases[search_type]

        # ========================================================
        # 2. Validate query
        # ========================================================

        query = str(query).strip()

        if not query:
            return "❌ GitHub search query cannot be empty."

        # ========================================================
        # 3. Validate result count
        # ========================================================

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5

        max_results = max(1, min(max_results, 10))

        # ========================================================
        # 4. GitHub API configuration
        # ========================================================

        api_url = f"https://api.github.com/search/{search_type}"

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "LangGraph-Chatbot",
        }

        # ========================================================
        # 5. GitHub authentication
        # ========================================================

        token = os.getenv("GITHUB_TOKEN")

        if token:
            headers["Authorization"] = f"Bearer {token}"

        params = {
            "q": query,
            "per_page": max_results,
        }

        # ========================================================
        # 6. Make API request
        # ========================================================

        response = requests.get(
            api_url,
            headers=headers,
            params=params,
            timeout=15,
        )

        # ========================================================
        # 7. Parse response safely
        # ========================================================

        try:
            data = response.json()
        except ValueError:
            data = {}

        # ========================================================
        # 8. Detailed HTTP error handling
        # ========================================================

        if response.status_code == 401:
            return (
                "❌ **GitHub authentication failed**\n\n"
                f"HTTP status: `{response.status_code}`\n\n"
                "Your GITHUB_TOKEN may be invalid, expired, "
                "or incorrectly configured in `.env`."
            )

        if response.status_code == 403:

            message = data.get(
                "message",
                "GitHub denied the request."
            )

            return (
                "❌ **GitHub API access denied**\n\n"
                f"HTTP status: `{response.status_code}`\n"
                f"GitHub message: `{message}`\n\n"
                "This may be caused by an API rate limit "
                "or insufficient authentication/permissions."
            )

        if response.status_code == 404:
            return (
                "❌ **GitHub API endpoint not found**\n\n"
                f"HTTP status: `{response.status_code}`\n"
                f"Endpoint: `{api_url}`"
            )

        if response.status_code == 422:

            message = data.get(
                "message",
                "GitHub rejected the search request."
            )

            errors = data.get("errors", [])

            error_details = ""

            if errors:
                error_details = (
                    "\n\nDetails:\n"
                    + "\n".join(
                        str(error)
                        for error in errors
                    )
                )

            return (
                "❌ **GitHub search request rejected**\n\n"
                f"HTTP status: `{response.status_code}`\n"
                f"GitHub message: `{message}`"
                f"{error_details}"
            )

        if response.status_code == 429:
            return (
                "❌ **GitHub API rate limit exceeded**\n\n"
                f"HTTP status: `{response.status_code}`\n\n"
                "Please wait and try again later."
            )

        if not response.ok:
            message = data.get(
                "message",
                "Unknown GitHub API error."
            )

            return (
                "❌ **GitHub API error**\n\n"
                f"HTTP status: `{response.status_code}`\n"
                f"Message: `{message}`"
            )

        # ========================================================
        # 9. Extract search results
        # ========================================================

        items = data.get("items", [])

        total = data.get(
            "total_count",
            len(items)
        )

        # ========================================================
        # 10. Important diagnostic for CODE search
        # ========================================================

        if search_type == "code" and not items:

            return (
                "🐙 **GitHub Code Search**\n\n"
                f"**Query:** `{query}`\n"
                f"**HTTP status:** `{response.status_code}`\n"
                f"**Total matches reported by GitHub:** `{total}`\n"
                f"**Results returned:** `0`\n\n"
                "⚠️ GitHub returned no code items for this "
                "request.\n\n"
                "This is different from the API request failing. "
                "The request itself completed successfully."
            )

        # ========================================================
        # 11. No results for other searches
        # ========================================================

        if not items:

            return (
                f"🐙 **GitHub {search_type.title()} Search**\n\n"
                f"**Query:** `{query}`\n"
                f"**HTTP status:** `{response.status_code}`\n\n"
                "No matching results were returned by GitHub."
            )

        # ========================================================
        # 12. Format results
        # ========================================================

        results = []

        for index, item in enumerate(
            items,
            start=1
        ):

            # ====================================================
            # REPOSITORIES
            # ====================================================

            if search_type == "repositories":

                repo_name = item.get(
                    "full_name",
                    "Unknown repository"
                )

                description = (
                    item.get("description")
                    or "No description"
                )

                language = (
                    item.get("language")
                    or "Unknown"
                )

                stars = item.get(
                    "stargazers_count",
                    0
                )

                forks = item.get(
                    "forks_count",
                    0
                )

                open_issues = item.get(
                    "open_issues_count",
                    0
                )

                html_url = item.get(
                    "html_url",
                    ""
                )

                results.append(
                    f"### {index}. ⭐ {repo_name}\n"
                    f"**Description:** {description}\n"
                    f"**Language:** {language}\n"
                    f"**Stars:** {stars:,}\n"
                    f"**Forks:** {forks:,}\n"
                    f"**Open issues:** {open_issues:,}\n"
                    f"🔗 {html_url}"
                )

            # ====================================================
            # CODE
            # ====================================================

            elif search_type == "code":

                repository = item.get(
                    "repository",
                    {}
                )

                repo_name = repository.get(
                    "full_name",
                    "Unknown repository"
                )

                path = item.get(
                    "path",
                    "Unknown file"
                )

                html_url = item.get(
                    "html_url",
                    ""
                )

                results.append(
                    f"### {index}. 💻 {repo_name}\n"
                    f"**File:** `{path}`\n"
                    f"🔗 {html_url}"
                )

            # ====================================================
            # ISSUES
            # ====================================================

            elif search_type == "issues":

                title = item.get(
                    "title",
                    "Untitled issue"
                )

                state = item.get(
                    "state",
                    "unknown"
                )

                comments = item.get(
                    "comments",
                    0
                )

                html_url = item.get(
                    "html_url",
                    ""
                )

                # GitHub search/issues also returns pull
                # requests. Detect them so the UI is clear.

                is_pull_request = bool(
                    item.get("pull_request")
                )

                item_type = (
                    "Pull Request"
                    if is_pull_request
                    else "Issue"
                )

                repository_url = item.get(
                    "repository_url",
                    ""
                )

                repository_name = (
                    repository_url
                    .replace(
                        "https://api.github.com/repos/",
                        ""
                    )
                    if repository_url
                    else "Unknown repository"
                )

                results.append(
                    f"### {index}. 🐛 {title}\n"
                    f"**Type:** {item_type}\n"
                    f"**Repository:** {repository_name}\n"
                    f"**State:** {state}\n"
                    f"**Comments:** {comments:,}\n"
                    f"🔗 {html_url}"
                )

            # ====================================================
            # USERS
            # ====================================================

            elif search_type == "users":

                login = item.get(
                    "login",
                    item.get(
                        "name",
                        "Unknown user"
                    )
                )

                user_type = item.get(
                    "type",
                    "User"
                )

                html_url = item.get(
                    "html_url",
                    ""
                )

                results.append(
                    f"### {index}. 👤 {login}\n"
                    f"**Type:** {user_type}\n"
                    f"🔗 {html_url}"
                )

        # ========================================================
        # 13. Final response
        # ========================================================

        return (
            f"🐙 **GitHub {search_type.title()} Search**\n\n"
            f"**Query:** `{query}`\n"
            f"**HTTP status:** `{response.status_code}`\n"
            f"**Results:** {len(items)} of "
            f"{total:,} total matches\n\n"
            + "\n\n".join(results)
        )

    # ============================================================
    # Network errors
    # ============================================================

    except requests.exceptions.Timeout:

        return (
            "❌ **GitHub search timed out.**\n\n"
            "GitHub did not respond within 15 seconds. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        return (
            "❌ **Could not connect to GitHub.**\n\n"
            "Check your internet connection and try again."
        )

    except requests.exceptions.RequestException as exc:

        return (
            f"❌ **GitHub request failed:**\n\n"
            f"`{exc}`"
        )

    except Exception as exc:

        return (
            f"❌ **Unexpected GitHub search error:**\n\n"
            f"`{exc}`"
        )
    

@tool
def lint_code(code: str, language: str = "python") -> str:
    """
    Lint source code and report syntax, style, and common Python issues.

    Args:
        code: The source code to analyze.
        language: Programming language. Currently supports Python.

    Returns:
        Human-readable linting results.
    """

    # ============================================================
    # 1. Validate language
    # ============================================================

    language = str(language).strip().lower()

    if language not in {"python", "py"}:
        return (
            f"❌ Linting for `{language}` is not currently supported.\n\n"
            "Supported language:\n"
            "• Python"
        )

    # ============================================================
    # 2. Validate code input
    # ============================================================

    if not isinstance(code, str) or not code.strip():
        return "❌ No code was provided to lint."

    tmp_path = None

    try:
        # ========================================================
        # 3. Create temporary Python file
        # ========================================================

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as temp_file:

            temp_file.write(code)
            tmp_path = temp_file.name

        # ========================================================
        # 4. First: syntax validation
        # ========================================================

        try:
            ast.parse(code)
        except SyntaxError as exc:

            line = exc.lineno or 0
            column = exc.offset or 0
            message = exc.msg or "Invalid syntax"

            source_line = (
                exc.text.strip()
                if exc.text
                else ""
            )

            pointer = ""

            if column > 0:
                pointer = " " * (column - 1) + "^"

            return (
                "❌ **Python syntax error**\n\n"
                f"**Line:** `{line}`\n"
                f"**Column:** `{column}`\n"
                f"**Error:** {message}\n\n"
                "```python\n"
                f"{source_line}\n"
                f"{pointer}\n"
                "```"
            )

        # ========================================================
        # 5. Check that Ruff is installed
        # ========================================================

        try:
            ruff_check = subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

        except FileNotFoundError:

            return (
                "⚠️ **Ruff is not installed.**\n\n"
                "The Python syntax is valid, but advanced "
                "linting cannot run.\n\n"
                "Install Ruff with:\n\n"
                "```bash\n"
                "pip install ruff\n"
                "```"
            )

        if ruff_check.returncode != 0:

            return (
                "⚠️ **Ruff could not be started.**\n\n"
                f"{ruff_check.stderr.strip()}"
            )

        # ========================================================
        # 6. Run Ruff
        # ========================================================

        result = subprocess.run(
            [
                "ruff",
                "check",
                tmp_path,
                "--output-format",
                "json"
            ],
            capture_output=True,
            text=True,
            timeout=15
        )

        # ========================================================
        # 7. Parse Ruff JSON
        # ========================================================

        try:
            diagnostics = json.loads(
                result.stdout
            ) if result.stdout.strip() else []

        except json.JSONDecodeError:

            return (
                "⚠️ **Ruff returned an unexpected response.**\n\n"
                "```text\n"
                f"{result.stdout.strip()}\n"
                f"{result.stderr.strip()}\n"
                "```"
            )

        # ========================================================
        # 8. No issues
        # ========================================================

        if not diagnostics:

            return (
                "✅ **Python code passed linting.**\n\n"
                "• Syntax: Valid\n"
                "• Ruff: No issues found\n"
                "• Style: OK\n"
                "• Common lint checks: OK"
            )

        # ========================================================
        # 9. Format diagnostics
        # ========================================================

        errors = 0
        warnings = 0

        formatted_results = []

        for index, diagnostic in enumerate(
            diagnostics,
            start=1
        ):

            code_id = diagnostic.get(
                "code",
                "UNKNOWN"
            )

            message = diagnostic.get(
                "message",
                "Unknown lint issue"
            )

            location = diagnostic.get(
                "location",
                {}
            )

            row = location.get(
                "row",
                "?"
            )

            column = location.get(
                "column",
                "?"
            )

            end_location = diagnostic.get(
                "end_location",
                {}
            )

            end_row = end_location.get(
                "row",
                row
            )

            end_column = end_location.get(
                "column",
                column
            )

            fix = diagnostic.get(
                "fix"
            )

            # Ruff's default output contains rule codes.
            # Treat E/F errors as errors and others as warnings.
            if code_id.startswith(("E", "F")):
                severity = "❌"
                errors += 1
            else:
                severity = "⚠️"
                warnings += 1

            location_text = (
                f"line {row}, column {column}"
            )

            if (
                end_row != row
                or end_column != column
            ):
                location_text += (
                    f" → line {end_row}, "
                    f"column {end_column}"
                )

            fix_text = ""

            if fix:
                fix_text = (
                    "\n"
                    "   🔧 **Automatic fix available**"
                )

            formatted_results.append(
                f"### {index}. {severity} `{code_id}`\n"
                f"**Location:** {location_text}\n"
                f"**Issue:** {message}"
                f"{fix_text}"
            )

        # ========================================================
        # 10. Final report
        # ========================================================

        return (
            "🐍 **Python Code Lint Report**\n\n"
            "### Summary\n"
            f"• Syntax: ✅ Valid\n"
            f"• Issues found: **{len(diagnostics)}**\n"
            f"• Errors: **{errors}**\n"
            f"• Warnings: **{warnings}**\n\n"
            + "\n\n".join(formatted_results)
        )

    # ============================================================
    # 11. Timeout
    # ============================================================

    except subprocess.TimeoutExpired:

        return (
            "❌ **Linting timed out.**\n\n"
            "Ruff did not finish within 15 seconds."
        )

    # ============================================================
    # 12. General errors
    # ============================================================

    except Exception as exc:

        return (
            "❌ **Linting error**\n\n"
            f"`{exc}`"
        )

    # ============================================================
    # 13. Always clean up temporary file
    # ============================================================

    finally:

        if tmp_path:

            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            except OSError:
                pass



# ============================================================
# FINANCE & CRYPTO TOOLS
# ============================================================

def _safe_float(value, default=None):
    """Convert a value to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value, decimals=2, prefix=""):
    """Format numeric values safely."""
    number = _safe_float(value)

    if number is None:
        return "N/A"

    return f"{prefix}{number:,.{decimals}f}"


def _format_market_cap(value):
    """Format large market-cap values."""
    number = _safe_float(value)

    if number is None or number == 0:
        return "N/A"

    if abs(number) >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f}T"

    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"

    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"

    return f"${number:,.0f}"


def _get_yfinance():
    """
    Import yfinance lazily and configure modest retries.
    """
    try:
        import yfinance as yf

        # Retry transient network failures.
        try:
            yf.config.network.retries = 2
        except Exception:
            pass

        return yf

    except ImportError:
        raise ImportError(
            "The 'yfinance' package is not installed. "
            "Install it with: pip install yfinance"
        )


# ============================================================
# 1. STOCK PRICE
# ============================================================

@tool
def get_stock_price(symbol: str) -> str:
    """
    Get current stock price and key market metrics.

    Args:
        symbol:
            Stock ticker symbol, e.g. AAPL, GOOGL, TSLA, MSFT.

    Returns:
        Current price, daily change, market cap, 52-week range,
        P/E ratio, volume, and company name.
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        return "❌ Please provide a stock ticker symbol."

    # Basic length protection.
    if len(symbol) > 20:
        return "❌ Invalid ticker symbol."

    try:
        yf = _get_yfinance()

        ticker = yf.Ticker(symbol)

        # --------------------------------------------------------
        # Price history
        # --------------------------------------------------------

        hist = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
            timeout=10
        )

        if hist.empty:
            return (
                f"❌ No market data found for `{symbol}`.\n\n"
                "Please check that the ticker symbol is correct."
            )

        closes = hist["Close"].dropna()

        if closes.empty:
            return (
                f"❌ No closing-price data available for `{symbol}`."
            )

        current = _safe_float(closes.iloc[-1])

        if current is None:
            return (
                f"❌ Could not determine the current price for `{symbol}`."
            )

        # --------------------------------------------------------
        # Previous close
        # --------------------------------------------------------

        previous = (
            _safe_float(closes.iloc[-2])
            if len(closes) >= 2
            else current
        )

        if previous is None or previous == 0:
            change = 0.0
            change_pct = 0.0
        else:
            change = current - previous
            change_pct = (change / previous) * 100

        direction = "📈" if change >= 0 else "📉"

        # --------------------------------------------------------
        # Company information
        # --------------------------------------------------------

        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or symbol
        )

        currency = (
            info.get("currency")
            or "USD"
        )

        market_cap = info.get("marketCap")
        week_high = info.get("fiftyTwoWeekHigh")
        week_low = info.get("fiftyTwoWeekLow")
        pe_ratio = info.get("trailingPE")
        volume = info.get("volume")

        # --------------------------------------------------------
        # Format output
        # --------------------------------------------------------

        return (
            f"{direction} **{company_name} ({symbol})**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Current Price:  "
            f"{currency} {current:,.2f}\n"
            f"• Daily Change:   "
            f"{change:+,.2f} ({change_pct:+.2f}%)\n"
            f"• Market Cap:     "
            f"{_format_market_cap(market_cap)}\n"
            f"• 52-Week High:   "
            f"{_format_number(week_high, 2, currency + ' ')}\n"
            f"• 52-Week Low:    "
            f"{_format_number(week_low, 2, currency + ' ')}\n"
            f"• P/E Ratio:      "
            f"{_format_number(pe_ratio, 2)}\n"
            f"• Volume:         "
            f"{_format_number(volume, 0)}\n"
            f"\n_Source: Yahoo Finance via yfinance._"
        )

    except ImportError as exc:
        return f"❌ {exc}"

    except Exception as exc:
        return (
            f"❌ **Stock price error for `{symbol}`**\n\n"
            f"`{exc}`"
        )


# ============================================================
# 2. CRYPTO PRICE
# ============================================================

@tool
def get_crypto_price(coin_id: str) -> str:
    """
    Get current cryptocurrency market data from CoinGecko.

    Args:
        coin_id:
            CoinGecko coin ID, e.g. bitcoin, ethereum, solana,
            dogecoin.

    Returns:
        Current price, 24h change, market cap, volume,
        24h high/low, ATH, and market rank.
    """

    coin_id = str(coin_id).strip().lower()

    if not coin_id:
        return "❌ Please provide a CoinGecko coin ID."

    # CoinGecko IDs normally use lowercase letters,
    # numbers and hyphens.
    if len(coin_id) > 100:
        return "❌ Invalid cryptocurrency ID."

    try:
        url = (
            "https://api.coingecko.com/api/v3/"
            "coins/markets"
        )

        params = {
            "vs_currency": "usd",
            "ids": coin_id,
            "price_change_percentage": "24h",
        }

        headers = {
            "Accept": "application/json",
            "User-Agent": "LangGraph-Finance-Tool/1.0",
        }

        # Optional CoinGecko Demo API key.
        #
        # If COINGECKO_API_KEY is not configured,
        # CoinGecko's keyless public API is used.
        api_key = os.getenv("COINGECKO_API_KEY")

        if api_key:
            headers["x-cg-demo-api-key"] = api_key

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15,
        )

        # --------------------------------------------------------
        # HTTP errors
        # --------------------------------------------------------

        if response.status_code == 401:
            return (
                "❌ **CoinGecko authentication failed.**\n\n"
                "Check your `COINGECKO_API_KEY`."
            )

        if response.status_code == 429:
            return (
                "❌ **CoinGecko rate limit reached.**\n\n"
                "Please wait and try again later."
            )

        if not response.ok:
            return (
                "❌ **CoinGecko API error**\n\n"
                f"HTTP status: `{response.status_code}`\n"
                f"Message: `{response.text[:500]}`"
            )

        data = response.json()

        if not isinstance(data, list) or not data:
            return (
                f"❌ No cryptocurrency found for "
                f"`{coin_id}`.\n\n"
                "Use the CoinGecko coin ID, for example:\n"
                "• bitcoin\n"
                "• ethereum\n"
                "• solana\n"
                "• dogecoin"
            )

        coin = data[0]

        name = coin.get("name") or coin_id
        symbol = str(
            coin.get("symbol") or ""
        ).upper()

        current_price = _safe_float(
            coin.get("current_price")
        )

        change_24h = _safe_float(
            coin.get("price_change_percentage_24h"),
            0.0
        )

        direction = (
            "📈"
            if change_24h >= 0
            else "📉"
        )

        return (
            f"{direction} **{name} ({symbol})**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Current Price: "
            f"${_format_number(current_price, 6)}\n"
            f"• 24h Change:    "
            f"{change_24h:+.2f}%\n"
            f"• Market Cap:    "
            f"{_format_market_cap(coin.get('market_cap'))}\n"
            f"• 24h Volume:    "
            f"{_format_market_cap(coin.get('total_volume'))}\n"
            f"• 24h High:      "
            f"${_format_number(coin.get('high_24h'), 6)}\n"
            f"• 24h Low:       "
            f"${_format_number(coin.get('low_24h'), 6)}\n"
            f"• All-Time High: "
            f"${_format_number(coin.get('ath'), 6)}\n"
            f"• Market Rank:   "
            f"#{coin.get('market_cap_rank', 'N/A')}\n"
            f"\n_Source: CoinGecko_"
        )

    except requests.exceptions.Timeout:
        return (
            "❌ CoinGecko request timed out. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌ Could not connect to CoinGecko. "
            "Check your internet connection."
        )

    except ValueError:
        return (
            "❌ CoinGecko returned invalid JSON data."
        )

    except Exception as exc:
        return (
            f"❌ **Crypto price error**\n\n"
            f"`{exc}`"
        )


# ============================================================
# 3. FOREX RATE
# ============================================================

@tool
def get_forex_rate(
    from_currency: str,
    to_currency: str
) -> str:
    """
    Get the latest available FX rate using Yahoo Finance.

    Args:
        from_currency:
            Three-letter currency code, e.g. USD, EUR, NPR.

        to_currency:
            Three-letter currency code, e.g. USD, EUR, INR, JPY.

    Returns:
        Latest available exchange rate and recent change.
    """

    from_currency = str(
        from_currency
    ).strip().upper()

    to_currency = str(
        to_currency
    ).strip().upper()

    if len(from_currency) != 3:
        return (
            f"❌ Invalid source currency: "
            f"`{from_currency}`"
        )

    if len(to_currency) != 3:
        return (
            f"❌ Invalid target currency: "
            f"`{to_currency}`"
        )

    if from_currency == to_currency:
        return (
            f"💱 **{from_currency} → {to_currency}**\n\n"
            f"1 {from_currency} = 1.0000 {to_currency}"
        )

    try:
        yf = _get_yfinance()

        pair = (
            f"{from_currency}"
            f"{to_currency}=X"
        )

        ticker = yf.Ticker(pair)

        hist = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
            timeout=10
        )

        if hist.empty:
            return (
                f"❌ Could not fetch FX data for "
                f"`{from_currency}/{to_currency}`."
            )

        closes = hist["Close"].dropna()

        if closes.empty:
            return (
                f"❌ No exchange-rate data available "
                f"for `{from_currency}/{to_currency}`."
            )

        rate = _safe_float(closes.iloc[-1])

        if rate is None:
            return "❌ Could not determine the exchange rate."

        previous = (
            _safe_float(closes.iloc[-2])
            if len(closes) >= 2
            else rate
        )

        if previous and previous != 0:
            change = rate - previous
            change_pct = (
                change / previous
            ) * 100
        else:
            change = 0.0
            change_pct = 0.0

        direction = (
            "📈"
            if change >= 0
            else "📉"
        )

        return (
            f"💱 **{from_currency} → {to_currency}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Rate:       "
            f"1 {from_currency} = "
            f"{rate:.6f} {to_currency}\n"
            f"• Change:     "
            f"{change:+.6f} ({change_pct:+.2f}%) {direction}\n"
            f"\n_Source: Yahoo Finance via yfinance._"
        )

    except ImportError as exc:
        return f"❌ {exc}"

    except Exception as exc:
        return (
            f"❌ **Forex rate error**\n\n"
            f"`{exc}`"
        )


# ============================================================
# 4. PORTFOLIO CALCULATOR
# ============================================================

@tool
def portfolio_calculator(holdings: str) -> str:
    """
    Calculate the current approximate value of stock/crypto holdings.

    Args:
        holdings:
            Comma-separated SYMBOL:QUANTITY values.

            Examples:
                AAPL:10,GOOGL:5
                AAPL:10,BTC-USD:0.5

            For cryptocurrency, use Yahoo Finance symbols such as
            BTC-USD and ETH-USD.

    Returns:
        Position-by-position values and total portfolio value.
    """

    if not isinstance(holdings, str):
        return "❌ Holdings must be provided as text."

    holdings = holdings.strip()

    if not holdings:
        return (
            "❌ No holdings provided.\n\n"
            "Example:\n"
            "`AAPL:10,GOOGL:5,BTC-USD:0.5`"
        )

    try:
        yf = _get_yfinance()

        items = [
            item.strip()
            for item in holdings.split(",")
            if item.strip()
        ]

        if not items:
            return "❌ No valid holdings found."

        positions = []
        total = 0.0

        for item in items:

            # ----------------------------------------------------
            # Validate format
            # ----------------------------------------------------

            if ":" not in item:
                positions.append({
                    "symbol": item,
                    "error": (
                        "Invalid format. "
                        "Use SYMBOL:QUANTITY"
                    )
                })
                continue

            symbol, qty_str = item.split(":", 1)

            symbol = symbol.strip().upper()
            qty_str = qty_str.strip()

            if not symbol:
                positions.append({
                    "symbol": "Unknown",
                    "error": "Missing symbol"
                })
                continue

            try:
                quantity = float(qty_str)
            except ValueError:
                positions.append({
                    "symbol": symbol,
                    "error": (
                        f"Invalid quantity `{qty_str}`"
                    )
                })
                continue

            if quantity <= 0:
                positions.append({
                    "symbol": symbol,
                    "error": (
                        "Quantity must be greater than zero"
                    )
                })
                continue

            # ----------------------------------------------------
            # Fetch price
            # ----------------------------------------------------

            try:
                ticker = yf.Ticker(symbol)

                hist = ticker.history(
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    timeout=10
                )

                if hist.empty:
                    positions.append({
                        "symbol": symbol,
                        "error": "No market data found"
                    })
                    continue

                closes = hist["Close"].dropna()

                if closes.empty:
                    positions.append({
                        "symbol": symbol,
                        "error": "No closing price available"
                    })
                    continue

                price = _safe_float(
                    closes.iloc[-1]
                )

                if price is None:
                    positions.append({
                        "symbol": symbol,
                        "error": "Invalid price returned"
                    })
                    continue

                value = price * quantity

                total += value

                positions.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "value": value,
                })

            except Exception as exc:

                positions.append({
                    "symbol": symbol,
                    "error": str(exc)
                })

        # --------------------------------------------------------
        # No successful positions
        # --------------------------------------------------------

        successful = [
            position
            for position in positions
            if "value" in position
        ]

        if not successful:

            return (
                "❌ **No holdings could be valued.**\n\n"
                "Check your ticker symbols and quantities.\n\n"
                "Example:\n"
                "`AAPL:10,MSFT:5,BTC-USD:0.5`"
            )

        # --------------------------------------------------------
        # Build report
        # --------------------------------------------------------

        rows = []

        for position in positions:

            symbol = position["symbol"]

            if "value" not in position:

                rows.append(
                    f"• **{symbol}**: ❌ "
                    f"{position['error']}"
                )

                continue

            quantity = position["quantity"]
            price = position["price"]
            value = position["value"]

            rows.append(
                f"• **{symbol}**: "
                f"{quantity:g} × "
                f"${price:,.2f} = "
                f"**${value:,.2f}**"
            )

        successful_value = sum(
            position["value"]
            for position in successful
        )

        return (
            "💼 **Portfolio Summary**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(rows)
            + "\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 **Total Value: "
            f"${successful_value:,.2f}**\n\n"
            "_Prices are latest available market data "
            "from Yahoo Finance via yfinance._"
        )

    except ImportError as exc:
        return f"❌ {exc}"

    except Exception as exc:
        return (
            f"❌ **Portfolio calculation error**\n\n"
            f"`{exc}`"
        )


# ============================================================
# ── NEW: FILE & DATA TOOLS ──────────────────────────────────
# ============================================================

@tool
def read_csv(filepath: str, max_rows: int = 20) -> str:
    """
    Read and analyze a CSV file.

    Provides:
    - File shape
    - Column names
    - Data types
    - Missing-value counts
    - Basic numeric statistics
    - Preview of rows

    Args:
        filepath: Path to the CSV file.
        max_rows: Number of preview rows, between 1 and 100.
    """
    try:
        import pandas as pd

        if not filepath or not filepath.strip():
            return "❌ Please provide a CSV file path."

        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            return "❌ max_rows must be an integer."

        max_rows = max(1, min(max_rows, 100))

        if not os.path.isfile(filepath):
            return f"❌ CSV file not found: `{filepath}`"

        # Try common encodings for real-world CSV files.
        df = None
        last_error = None

        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc

        if df is None:
            return (
                "❌ Could not decode the CSV file.\n\n"
                f"Last encoding error: `{last_error}`"
            )

        rows, columns = df.shape

        # Column information
        column_info = "\n".join(
            f"• **{column}** → {dtype}"
            for column, dtype in df.dtypes.items()
        )

        # Missing values
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if missing.empty:
            missing_info = "None"
        else:
            missing_info = "\n".join(
                f"• **{column}**: {count:,}"
                for column, count in missing.items()
            )

        # Numeric statistics
        numeric_df = df.select_dtypes(include="number")

        if not numeric_df.empty:
            stats = numeric_df.describe().round(2).to_string()
            statistics = f"```text\n{stats}\n```"
        else:
            statistics = "No numeric columns available."

        # Preview
        preview = df.head(max_rows).to_string(index=False)

        return (
            f"📊 **CSV Analysis: {os.path.basename(filepath)}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Rows:       {rows:,}\n"
            f"• Columns:    {columns:,}\n"
            f"• File Size:  "
            f"{os.path.getsize(filepath) / 1024:.1f} KB\n\n"

            f"**Column Types**\n"
            f"{column_info}\n\n"

            f"**Missing Values**\n"
            f"{missing_info}\n\n"

            f"**Numeric Statistics**\n"
            f"{statistics}\n\n"

            f"**Preview — first {min(max_rows, rows)} rows**\n"
            f"```text\n{preview}\n```"
        )

    except ImportError:
        return (
            "❌ The 'pandas' package is not installed.\n"
            "Run: `python -m pip install pandas`"
        )

    except pd.errors.EmptyDataError:
        return "❌ The CSV file is empty."

    except pd.errors.ParserError as exc:
        return (
            "❌ Could not parse the CSV file.\n\n"
            f"Parser error: `{exc}`"
        )

    except PermissionError:
        return f"❌ Permission denied when reading `{filepath}`."

    except Exception as exc:
        return (
            f"❌ **CSV read error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


@tool
def extract_pdf_text(filepath: str, max_pages: int = 5) -> str:
    """
    Extract text from a PDF document.

    Args:
        filepath: Path to the PDF file.
        max_pages: Maximum number of pages to extract.
                    Between 1 and 20.

    Returns:
        Extracted text with page boundaries.
    """
    try:
        import pdfplumber

        if not filepath or not filepath.strip():
            return "❌ Please provide a PDF file path."

        try:
            max_pages = int(max_pages)
        except (TypeError, ValueError):
            return "❌ max_pages must be an integer."

        max_pages = max(1, min(max_pages, 20))

        if not os.path.isfile(filepath):
            return f"❌ PDF file not found: `{filepath}`"

        with pdfplumber.open(filepath) as pdf:

            total_pages = len(pdf.pages)

            if total_pages == 0:
                return "❌ The PDF contains no pages."

            pages_to_read = min(max_pages, total_pages)

            text_parts = []
            empty_pages = []

            for page_number in range(pages_to_read):
                page = pdf.pages[page_number]

                try:
                    text = page.extract_text(
                        x_tolerance=2,
                        y_tolerance=3
                    ) or ""
                except Exception as page_error:
                    text = ""
                    empty_pages.append(
                        f"Page {page_number + 1}: {page_error}"
                    )

                text = text.strip()

                if not text:
                    empty_pages.append(
                        f"Page {page_number + 1}: no extractable text"
                    )

                text_parts.append(
                    f"--- Page {page_number + 1} ---\n"
                    f"{text if text else '[No extractable text]'}"
                )

            full_text = "\n\n".join(text_parts)

            # Prevent enormous tool responses.
            max_chars = 12000

            truncated = False

            if len(full_text) > max_chars:
                full_text = (
                    full_text[:max_chars]
                    + "\n\n… [Output truncated]"
                )
                truncated = True

            result = (
                f"📄 **PDF: {os.path.basename(filepath)}**\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Total pages:    {total_pages}\n"
                f"• Pages read:     {pages_to_read}\n"
                f"• File size:      "
                f"{os.path.getsize(filepath) / 1024:.1f} KB\n\n"
                f"{full_text}"
            )

            if total_pages > pages_to_read:
                result += (
                    f"\n\n⚠️ Only the first {pages_to_read} "
                    f"of {total_pages} pages were extracted."
                )

            if empty_pages:
                result += (
                    "\n\n⚠️ **Extraction notes:**\n"
                    + "\n".join(
                        f"• {item}"
                        for item in empty_pages[:10]
                    )
                )

                result += (
                    "\n\nThis may indicate a scanned/image-only "
                    "PDF that requires OCR."
                )

            if truncated:
                result += (
                    "\n\n⚠️ The extracted text was truncated "
                    "to keep the tool response manageable."
                )

            return result

    except ImportError:
        return (
            "❌ The 'pdfplumber' package is not installed.\n"
            "Run: `python -m pip install pdfplumber`"
        )

    except PermissionError:
        return f"❌ Permission denied when reading `{filepath}`."

    except Exception as exc:
        return (
            f"❌ **PDF extraction error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


@tool
def read_docx(filepath: str) -> str:
    """
    Read text and tables from a Microsoft Word DOCX document.

    Args:
        filepath: Path to the .docx file.

    Returns:
        Document paragraphs, tables, and basic structure.
    """
    try:
        from docx import Document

        if not filepath or not filepath.strip():
            return "❌ Please provide a DOCX file path."

        if not os.path.isfile(filepath):
            return f"❌ Word document not found: `{filepath}`"

        doc = Document(filepath)

        sections = []

        # ----------------------------------------------------
        # Paragraphs
        # ----------------------------------------------------
        paragraphs = [
            p.text.strip()
            for p in doc.paragraphs
            if p.text.strip()
        ]

        if paragraphs:
            paragraph_text = "\n\n".join(paragraphs)

            if len(paragraph_text) > 8000:
                paragraph_text = (
                    paragraph_text[:8000]
                    + "\n\n… [Paragraph text truncated]"
                )

            sections.append(
                f"**Document Text**\n\n{paragraph_text}"
            )

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------
        table_sections = []

        for table_index, table in enumerate(doc.tables, start=1):

            rows = []

            for row in table.rows:
                cells = [
                    cell.text.replace("\n", " ").strip()
                    for cell in row.cells
                ]

                rows.append(" | ".join(cells))

            if rows:
                table_sections.append(
                    f"**Table {table_index}**\n"
                    + "\n".join(rows)
                )

        if table_sections:
            sections.append(
                "**Tables**\n\n"
                + "\n\n".join(table_sections)
            )

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------
        if not sections:
            content = "[Document contains no readable text.]"
        else:
            content = "\n\n".join(sections)

        max_chars = 12000

        if len(content) > max_chars:
            content = (
                content[:max_chars]
                + "\n\n… [Output truncated]"
            )

        return (
            f"📝 **Word Document: "
            f"{os.path.basename(filepath)}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Paragraphs: {len(paragraphs)}\n"
            f"• Tables:     {len(doc.tables)}\n"
            f"• File Size:  "
            f"{os.path.getsize(filepath) / 1024:.1f} KB\n\n"
            f"{content}"
        )

    except ImportError:
        return (
            "❌ The 'python-docx' package is not installed.\n"
            "Run: `python -m pip install python-docx`"
        )

    except PermissionError:
        return f"❌ Permission denied when reading `{filepath}`."

    except Exception as exc:
        return (
            f"❌ **DOCX read error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


@tool
def query_sql(database_path: str, sql_query: str) -> str:
    """
    Execute a read-only SQL query against a SQLite database.

    Only SELECT and WITH (CTE) queries are allowed.
    INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA,
    ATTACH, DETACH, and other modifying statements are rejected.

    Args:
        database_path: Path to the SQLite database.
        sql_query: SQL SELECT/WITH query.

    Returns:
        Formatted query results.
    """
    db_conn = None

    try:
        if not database_path or not database_path.strip():
            return "❌ Please provide a database path."

        if not sql_query or not sql_query.strip():
            return "❌ Please provide a SQL query."

        if not os.path.isfile(database_path):
            return f"❌ Database not found: `{database_path}`"

        query = sql_query.strip()

        # Remove trailing semicolon for easier validation.
        normalized = query.rstrip(";").strip()

        if not normalized:
            return "❌ SQL query is empty."

        # ----------------------------------------------------
        # Safety validation
        # ----------------------------------------------------

        # Only one SQL statement is permitted.
        statements = [
            statement.strip()
            for statement in normalized.split(";")
            if statement.strip()
        ]

        if len(statements) != 1:
            return (
                "❌ Multiple SQL statements are not allowed.\n\n"
                "Only one read-only query may be executed."
            )

        first_keyword = (
            normalized.split(None, 1)[0].upper()
            if normalized.split()
            else ""
        )

        # SELECT and WITH are the only allowed entry points.
        if first_keyword not in {"SELECT", "WITH"}:
            return (
                "❌ **Read-only SQL restriction**\n\n"
                "Only `SELECT` and `WITH` queries are allowed."
            )

        # Block dangerous SQLite statements/keywords.
        blocked_keywords = {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "REPLACE",
            "UPSERT",
            "ATTACH",
            "DETACH",
            "VACUUM",
            "REINDEX",
            "PRAGMA",
        }

        import re

        upper_query = normalized.upper()

        for keyword in blocked_keywords:
            if re.search(
                rf"\b{re.escape(keyword)}\b",
                upper_query
            ):
                return (
                    f"❌ SQL statement contains blocked "
                    f"operation: `{keyword}`\n\n"
                    "This tool is read-only."
                )

        # ----------------------------------------------------
        # Execute query
        # ----------------------------------------------------

        db_conn = sqlite3.connect(
            database_path,
            check_same_thread=False,
            timeout=10
        )

        # Read-only SQLite connection where possible.
        db_conn.execute("PRAGMA query_only = ON")

        cursor = db_conn.cursor()

        cursor.execute(normalized)

        rows = cursor.fetchmany(100)

        columns = [
            description[0]
            for description in cursor.description
        ] if cursor.description else []

        # Check whether more rows exist.
        more_rows = cursor.fetchone() is not None

        if not rows:
            return (
                "🗄️ **SQL Query Result**\n\n"
                "Query executed successfully, but returned "
                "no rows."
            )

        # ----------------------------------------------------
        # Format table
        # ----------------------------------------------------

        def format_value(value):
            if value is None:
                return "NULL"

            text = str(value)

            # Prevent extremely large individual cells.
            if len(text) > 200:
                text = text[:200] + "…"

            return text.replace("\n", " ")

        formatted_rows = [
            [format_value(value) for value in row]
            for row in rows
        ]

        # Calculate column widths.
        widths = []

        for index, column in enumerate(columns):
            column_width = len(str(column))

            for row in formatted_rows:
                if index < len(row):
                    column_width = max(
                        column_width,
                        len(row[index])
                    )

            widths.append(min(column_width, 40))

        header = " | ".join(
            str(column)[:40].ljust(widths[index])
            for index, column in enumerate(columns)
        )

        separator = "-+-".join(
            "-" * width
            for width in widths
        )

        body_rows = []

        for row in formatted_rows:
            body_rows.append(
                " | ".join(
                    row[index][:40].ljust(widths[index])
                    for index in range(len(columns))
                )
            )

        body = "\n".join(body_rows)

        result = (
            f"🗄️ **SQL Query Result**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Database: `{os.path.basename(database_path)}`\n\n"
            f"```text\n"
            f"{header}\n"
            f"{separator}\n"
            f"{body}\n"
            f"```"
        )

        if more_rows:
            result += (
                "\n\n⚠️ Showing the first **100 rows**."
            )
        else:
            result += (
                f"\n\nShowing **{len(rows)} row(s)**."
            )

        return result

    except sqlite3.OperationalError as exc:
        return (
            "❌ **SQL execution error**\n\n"
            f"`{exc}`"
        )

    except sqlite3.DatabaseError as exc:
        return (
            "❌ **SQLite database error**\n\n"
            f"`{exc}`"
        )

    except PermissionError:
        return (
            f"❌ Permission denied accessing "
            f"`{database_path}`."
        )

    except Exception as exc:
        return (
            f"❌ **SQL query error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )

    finally:
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass


# ============================================================
# ── NEW: LOCATION & MAPS TOOLS ──────────────────────────────
# ============================================================

import ipaddress


@tool
def geocode_address(address: str) -> str:
    """
    Convert an address, city, landmark, or place name into
    geographic coordinates using the Open-Meteo Geocoding API.

    Args:
        address:
            Address or place name, e.g.:
            'Eiffel Tower'
            'Sydney, Australia'
            'Kathmandu, Nepal'

    Returns:
        Up to 3 matching locations with coordinates,
        country, administrative region, timezone, and population.
    """
    try:
        if not isinstance(address, str):
            return "❌ Address must be provided as text."

        address = address.strip()

        if not address:
            return "❌ Please provide an address or place name."

        if len(address) > 200:
            return "❌ Address is too long. Please provide a shorter location."

        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": address,
                "count": 3,
                "language": "en",
                "format": "json",
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "LangGraph-Chatbot/1.0",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            return (
                f"❌ Could not geocode **{address}**.\n\n"
                "Try a more specific location, such as:\n"
                "• `Kathmandu, Nepal`\n"
                "• `Sydney, Australia`\n"
                "• `Eiffel Tower, Paris`"
            )

        output = [
            f"📍 **Geocoding Results**",
            f"Query: `{address}`",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        for index, loc in enumerate(results[:3], start=1):

            name = loc.get("name") or "Unknown"
            country = loc.get("country") or "Unknown"
            country_code = loc.get("country_code") or ""

            admin1 = loc.get("admin1") or "N/A"
            admin2 = loc.get("admin2") or "N/A"

            latitude = loc.get("latitude")
            longitude = loc.get("longitude")

            timezone = loc.get("timezone") or "N/A"
            population = loc.get("population")

            population_text = (
                f"{population:,.0f}"
                if isinstance(population, (int, float))
                else "N/A"
            )

            output.append(
                f"\n**{index}. {name}, {country} "
                f"({country_code})**\n"
                f"• Latitude:    {latitude}\n"
                f"• Longitude:   {longitude}\n"
                f"• Region:      {admin1}\n"
                f"• Subregion:   {admin2}\n"
                f"• Timezone:    {timezone}\n"
                f"• Population:  {population_text}"
            )

        output.append(
            "\n\n_Source: Open-Meteo Geocoding API_"
        )

        return "\n".join(output)

    except requests.exceptions.Timeout:
        return (
            "❌ **Geocoding request timed out.**\n\n"
            "Please try again."
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌ **Could not connect to the geocoding service.**\n\n"
            "Check your internet connection and try again."
        )

    except requests.exceptions.HTTPError as exc:
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        return (
            f"❌ **Geocoding API error**\n\n"
            f"HTTP status: `{status}`"
        )

    except ValueError:
        return (
            "❌ The geocoding service returned "
            "invalid JSON data."
        )

    except Exception as exc:
        return (
            f"❌ **Geocoding error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


@tool
def get_timezone(location: str) -> str:
    """
    Get the timezone and current local time for a city or location.

    Args:
        location:
            City or place name, e.g.:
            'Tokyo'
            'New York'
            'Kathmandu, Nepal'

    Returns:
        Timezone, current local time, UTC offset, and coordinates.
    """
    try:
        if not isinstance(location, str):
            return "❌ Location must be provided as text."

        location = location.strip()

        if not location:
            return "❌ Please provide a city or location."

        if len(location) > 200:
            return "❌ Location name is too long."

        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "LangGraph-Chatbot/1.0",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            return (
                f"❌ Could not find **{location}**.\n\n"
                "Try including the country, for example:\n"
                "`Kathmandu, Nepal`"
            )

        loc = results[0]

        name = loc.get("name") or location
        country = loc.get("country") or "Unknown"
        country_code = loc.get("country_code") or ""

        timezone_name = loc.get("timezone")

        latitude = loc.get("latitude")
        longitude = loc.get("longitude")

        if not timezone_name:
            return (
                f"❌ No timezone information was returned "
                f"for **{name}**."
            )

        try:
            tz = ZoneInfo(timezone_name)

            now = datetime.now(tz)

            local_time = now.strftime(
                "%A, %B %d, %Y — %I:%M:%S %p"
            )

            utc_offset = now.strftime("%z")

            if len(utc_offset) == 5:
                utc_offset = (
                    utc_offset[:3]
                    + ":"
                    + utc_offset[3:]
                )

            iso_time = now.isoformat()

        except Exception as exc:
            return (
                f"❌ Could not calculate local time for "
                f"timezone `{timezone_name}`.\n\n"
                f"`{exc}`"
            )

        return (
            f"🕐 **Local Time — {name}, {country} "
            f"({country_code})**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Timezone:    `{timezone_name}`\n"
            f"• Local Time:  {local_time}\n"
            f"• UTC Offset:  UTC{utc_offset}\n"
            f"• Coordinates: {latitude}, {longitude}\n"
            f"• ISO Time:    `{iso_time}`\n\n"
            f"_Source: Open-Meteo Geocoding API + Python ZoneInfo_"
        )

    except requests.exceptions.Timeout:
        return (
            "❌ **Timezone lookup timed out.**\n\n"
            "Please try again."
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌ **Could not connect to the geocoding service.**\n\n"
            "Check your internet connection and try again."
        )

    except requests.exceptions.HTTPError as exc:
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        return (
            f"❌ **Timezone API error**\n\n"
            f"HTTP status: `{status}`"
        )

    except ValueError:
        return (
            "❌ The geocoding service returned "
            "invalid JSON data."
        )

    except Exception as exc:
        return (
            f"❌ **Timezone lookup error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


@tool
def ip_geolocation(ip_address: str = "") -> str:
    """
    Get approximate geographic information for an IP address.

    If ip_address is blank, the public IP of the machine running
    the chatbot is used.

    Args:
        ip_address:
            IPv4 or IPv6 address.
            Leave blank to detect the current public IP.

    Returns:
        Country, region, city, coordinates, timezone, ISP,
        organization, and approximate location information.
    """
    try:
        if not isinstance(ip_address, str):
            return "❌ IP address must be provided as text."

        target = ip_address.strip()

        # ----------------------------------------------------
        # Validate explicit IP addresses
        # ----------------------------------------------------

        if target:

            try:
                ipaddress.ip_address(target)

            except ValueError:
                return (
                    f"❌ Invalid IP address: `{target}`\n\n"
                    "Provide a valid IPv4 or IPv6 address."
                )

        # HTTPS instead of HTTP.
        if target:
            url = (
                f"https://ip-api.com/json/"
                f"{target}"
            )
        else:
            url = "https://ip-api.com/json/"

        response = requests.get(
            url,
            params={
                "fields": (
                    "status,message,query,country,countryCode,"
                    "region,regionName,city,zip,lat,lon,timezone,"
                    "isp,org,as,asname"
                )
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "LangGraph-Chatbot/1.0",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            message = data.get(
                "message",
                "Unknown geolocation error."
            )

            return (
                "❌ **IP geolocation failed**\n\n"
                f"Reason: `{message}`"
            )

        query_ip = data.get("query") or target or "Unknown"

        country = data.get("country") or "N/A"
        country_code = data.get("countryCode") or "N/A"

        region = data.get("regionName") or "N/A"
        city = data.get("city") or "N/A"
        postal_code = data.get("zip") or "N/A"

        latitude = data.get("lat")
        longitude = data.get("lon")

        timezone_name = data.get("timezone") or "N/A"

        isp = data.get("isp") or "N/A"
        organization = data.get("org") or "N/A"

        as_number = data.get("as") or "N/A"
        as_name = data.get("asname") or "N/A"

        return (
            f"🌐 **IP Geolocation**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• IP Address:   `{query_ip}`\n"
            f"• Country:      {country} ({country_code})\n"
            f"• Region:       {region}\n"
            f"• City:         {city}\n"
            f"• Postal Code:  {postal_code}\n"
            f"• Latitude:     {latitude}\n"
            f"• Longitude:    {longitude}\n"
            f"• Timezone:     {timezone_name}\n"
            f"• ISP:          {isp}\n"
            f"• Organization: {organization}\n"
            f"• ASN:          {as_number}\n"
            f"• AS Name:      {as_name}\n\n"
            f"⚠️ IP geolocation is approximate and "
            f"should not be treated as an exact physical address.\n\n"
            f"_Source: ip-api.com_"
        )

    except requests.exceptions.Timeout:
        return (
            "❌ **IP geolocation request timed out.**\n\n"
            "Please try again."
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌ **Could not connect to the IP geolocation service.**\n\n"
            "Check your internet connection and try again."
        )

    except requests.exceptions.HTTPError as exc:
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        return (
            f"❌ **IP geolocation API error**\n\n"
            f"HTTP status: `{status}`"
        )

    except ValueError:
        return (
            "❌ The IP geolocation service returned "
            "invalid JSON data."
        )

    except Exception as exc:
        return (
            f"❌ **IP geolocation error**\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )


# ============================================================
# ── AI MEMORY TOOLS ─────────────────────────────────────────
# ============================================================

def _clean_memory_key(key: str) -> str:
    """Normalize and validate a memory key."""
    if not isinstance(key, str):
        raise ValueError("Memory key must be a string.")

    key = key.strip().lower()

    if not key:
        raise ValueError("Memory key cannot be empty.")

    if len(key) > 100:
        raise ValueError("Memory key must be 100 characters or fewer.")

    return key


def _clean_memory_value(value: str) -> str:
    """Validate and normalize a memory value."""
    if not isinstance(value, str):
        raise ValueError("Memory value must be a string.")

    value = value.strip()

    if not value:
        raise ValueError("Memory value cannot be empty.")

    if len(value) > 2000:
        raise ValueError("Memory value must be 2,000 characters or fewer.")

    return value


@tool
def remember_fact(key: str, value: str) -> str:
    """
    Store or update a fact in persistent SQLite memory.

    Use this when the user explicitly asks the assistant to remember
    something for future conversations.

    Args:
        key: Short descriptive label such as 'user_name',
             'favorite_color', or 'project_goal'.
        value: Information to store.
    """
    try:
        key = _clean_memory_key(key)
        value = _clean_memory_value(value)

        cursor = conn.cursor()

        # Check whether the memory already exists.
        cursor.execute(
            "SELECT value FROM memories WHERE key = ?",
            (key,)
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                """
                UPDATE memories
                SET value = ?, created_at = CURRENT_TIMESTAMP
                WHERE key = ?
                """,
                (value, key)
            )
            action = "Updated"
        else:
            cursor.execute(
                """
                INSERT INTO memories (key, value)
                VALUES (?, ?)
                """,
                (key, value)
            )
            action = "Remembered"

        conn.commit()

        return f'🧠 **{action}:** `{key}` → "{value}"'

    except ValueError as exc:
        return f"⚠️ Invalid memory: {exc}"

    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

        return "❌ Memory store error. The memory could not be saved."


@tool
def recall_memories(
    query: str = "",
    max_results: int = 20
) -> str:
    """
    Recall stored memories.

    Optionally filter memories by keyword in either the key or value.

    Args:
        query: Optional keyword or phrase to search for.
        max_results: Maximum number of memories to return (1-50).
    """
    try:
        if not isinstance(query, str):
            return "⚠️ Memory search query must be text."

        query = query.strip()

        if not isinstance(max_results, int):
            return "⚠️ max_results must be an integer."

        max_results = max(1, min(max_results, 50))

        cursor = conn.cursor()

        if query:
            search_pattern = f"%{query}%"

            cursor.execute(
                """
                SELECT key, value, created_at
                FROM memories
                WHERE key LIKE ? OR value LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (search_pattern, search_pattern, max_results)
            )
        else:
            cursor.execute(
                """
                SELECT key, value, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max_results,)
            )

        rows = cursor.fetchall()

        if not rows:
            if query:
                return f"🧠 No memories found matching '{query}'."
            return "🧠 No memories stored yet."

        lines = [
            f"🧠 **Stored Memories** ({len(rows)} shown):"
        ]

        for key, value, created_at in rows:
            saved_date = str(created_at)[:10] if created_at else "unknown"

            lines.append(
                f"• **{key}**: {value} "
                f"*(saved {saved_date})*"
            )

        return "\n".join(lines)

    except Exception:
        return "❌ Memory recall error. The memories could not be retrieved."


@tool
def forget_memory(key: str) -> str:
    """
    Delete a specific stored memory by its key.

    Use this when the user explicitly asks the assistant to forget
    or delete a particular remembered fact.

    Args:
        key: The key of the memory to delete.
    """
    try:
        key = _clean_memory_key(key)

        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM memories WHERE key = ?",
            (key,)
        )

        deleted = cursor.rowcount

        conn.commit()

        if deleted:
            return f"🗑️ **Forgotten:** `{key}`"

        return f"🧠 No memory found with key `{key}`."

    except ValueError as exc:
        return f"⚠️ Invalid memory key: {exc}"

    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

        return "❌ Memory delete error. The memory could not be deleted."


@tool
def search_past_conversations(
    keyword: str,
    max_results: int = 5
) -> str:
    """
    Search past conversation titles for a keyword.

    This searches conversation metadata/titles, not the full
    conversation message contents.

    Args:
        keyword: Word or phrase to search for in conversation titles.
        max_results: Maximum number of results to return (1-20).
    """
    try:
        if not isinstance(keyword, str):
            return "⚠️ Search keyword must be text."

        keyword = keyword.strip()

        if not keyword:
            return "⚠️ Please provide a keyword to search for."

        if len(keyword) > 200:
            return "⚠️ Search keyword must be 200 characters or fewer."

        if not isinstance(max_results, int):
            return "⚠️ max_results must be an integer."

        max_results = max(1, min(max_results, 20))

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT thread_id, title, updated_at
            FROM conversations
            WHERE title LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (f"%{keyword}%", max_results)
        )

        rows = cursor.fetchall()

        if not rows:
            return (
                f"🔍 No past conversations found "
                f"with '{keyword}' in the title."
            )

        lines = [
            f"🔍 **Past Conversations matching '{keyword}'** "
            f"({len(rows)} found):"
        ]

        for thread_id, title, updated_at in rows:
            title = title or "(Untitled)"
            date = str(updated_at)[:10] if updated_at else "unknown"

            lines.append(
                f"• **{title}** — last active {date}"
            )

        return "\n".join(lines)

    except Exception:
        return "❌ Conversation search error. Past conversations could not be searched."




# ============================================================
# ── IMAGE TOOLS ─────────────────────────────────────────────
# ============================================================

@tool
def resize_image(
    filepath: str,
    width: int,
    height: int,
    output_path: str = ""
) -> str:
    """
    Resize an image to exact pixel dimensions.

    Supports common formats such as JPG, JPEG, PNG, WEBP, BMP,
    GIF, and TIFF when supported by Pillow.

    Args:
        filepath: Path to the source image.
        width: Target width in pixels (1-8000).
        height: Target height in pixels (1-8000).
        output_path: Optional destination path. If omitted,
                     creates '<name>_resized<extension>'.
    """
    try:
        from PIL import Image, UnidentifiedImageError

        # -----------------------------
        # Validate input
        # -----------------------------
        if not isinstance(filepath, str) or not filepath.strip():
            return "⚠️ Image filepath cannot be empty."

        filepath = filepath.strip()

        if not os.path.isfile(filepath):
            return f"❌ Image file not found: {filepath}"

        if not isinstance(width, int) or not isinstance(height, int):
            return "⚠️ Width and height must be integers."

        if width < 1 or height < 1:
            return "⚠️ Width and height must be greater than 0."

        if width > 8000 or height > 8000:
            return "⚠️ Maximum supported dimension is 8000×8000 pixels."

        # Prevent accidentally creating enormous images.
        if width * height > 64_000_000:
            return "⚠️ Target image is too large. Maximum area is 64 million pixels."

        # -----------------------------
        # Determine output path
        # -----------------------------
        if output_path:
            output_path = output_path.strip()

            if not output_path:
                output_path = ""

        if not output_path:
            base, ext = os.path.splitext(filepath)
            output_path = f"{base}_resized{ext}"

        output_dir = os.path.dirname(os.path.abspath(output_path))

        if not os.path.isdir(output_dir):
            return f"❌ Output directory does not exist: {output_dir}"

        # Avoid overwriting the original image.
        if os.path.abspath(filepath) == os.path.abspath(output_path):
            return "⚠️ Output path must be different from the source image."

        # -----------------------------
        # Open and validate image
        # -----------------------------
        try:
            with Image.open(filepath) as img:
                img.verify()

            # Re-open after verify(); verify() invalidates the image object.
            with Image.open(filepath) as img:
                original_size = img.size
                original_format = img.format
                original_mode = img.mode

                # Load image data before closing.
                img.load()

                resized = img.resize(
                    (width, height),
                    Image.Resampling.LANCZOS
                )

                # -----------------------------
                # Preserve format compatibility
                # -----------------------------
                save_kwargs = {}

                output_ext = os.path.splitext(output_path)[1].lower()

                # JPEG does not support RGBA/P modes.
                if output_ext in {".jpg", ".jpeg"}:
                    if resized.mode in {"RGBA", "LA", "P"}:
                        # Preserve transparency as much as possible by
                        # compositing onto white for JPEG output.
                        if resized.mode in {"RGBA", "LA"}:
                            background = Image.new("RGB", resized.size, "white")
                            if resized.mode == "RGBA":
                                background.paste(
                                    resized,
                                    mask=resized.getchannel("A")
                                )
                            else:
                                background.paste(
                                    resized,
                                    mask=resized.getchannel("A")
                                )
                            resized = background
                        else:
                            resized = resized.convert("RGB")

                    save_kwargs["quality"] = 95
                    save_kwargs["optimize"] = True

                elif output_ext == ".png":
                    save_kwargs["optimize"] = True

                elif output_ext == ".webp":
                    save_kwargs["quality"] = 95
                    save_kwargs["method"] = 6

                # Save.
                resized.save(output_path, **save_kwargs)

                new_file_size = os.path.getsize(output_path)

        except UnidentifiedImageError:
            return (
                "❌ The file is not a valid or supported image. "
                "It may be corrupted or use an unsupported format."
            )

        except OSError as exc:
            return f"❌ Could not process image: {exc}"

        return (
            "✅ **Image resized successfully**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Original:    {original_size[0]}×{original_size[1]} px\n"
            f"• New size:    {width}×{height} px\n"
            f"• Format:      {original_format or 'Unknown'}\n"
            f"• Mode:        {original_mode}\n"
            f"• Output:      {output_path}\n"
            f"• Output size: {new_file_size / 1024:.1f} KB"
        )

    except ImportError:
        return (
            "❌ Pillow is not installed.\n"
            "Run: python -m pip install Pillow"
        )

    except Exception:
        return (
            "❌ Image resize error. "
            "The image could not be resized."
        )


@tool
def get_image_info(filepath: str) -> str:
    """
    Get detailed metadata and technical information about an image.

    Args:
        filepath: Path to the image file.
    """
    try:
        from PIL import Image, UnidentifiedImageError

        # -----------------------------
        # Validate input
        # -----------------------------
        if not isinstance(filepath, str) or not filepath.strip():
            return "⚠️ Image filepath cannot be empty."

        filepath = filepath.strip()

        if not os.path.isfile(filepath):
            return f"❌ Image file not found: {filepath}"

        file_size = os.path.getsize(filepath)

        if file_size == 0:
            return "❌ Image file is empty."

        # -----------------------------
        # Open and validate image
        # -----------------------------
        try:
            with Image.open(filepath) as img:
                # Force Pillow to read image data so corrupted files
                # are detected rather than just reading headers.
                img.load()

                filename = os.path.basename(filepath)
                format_name = img.format or "Unknown"
                mode = img.mode
                width, height = img.size

                # Number of channels where meaningful.
                channels = {
                    "1": 1,
                    "L": 1,
                    "LA": 2,
                    "RGB": 3,
                    "RGBA": 4,
                    "CMYK": 4,
                    "YCbCr": 3,
                    "P": 1,
                    "I": 1,
                    "F": 1,
                }.get(mode, "Unknown")

                megapixels = (width * height) / 1_000_000

                has_alpha = "A" in img.getbands()

                # EXIF information count only; do not dump potentially
                # sensitive metadata into the response.
                try:
                    exif = img.getexif()
                    exif_count = len(exif) if exif else 0
                except Exception:
                    exif_count = 0

        except UnidentifiedImageError:
            return (
                "❌ The file is not a valid or supported image. "
                "It may be corrupted or unsupported."
            )

        except OSError as exc:
            return f"❌ Could not read image: {exc}"

        # -----------------------------
        # Human-readable result
        # -----------------------------
        return (
            f"🖼️ **Image Information: {filename}**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Format:       {format_name}\n"
            f"• Mode:         {mode}\n"
            f"• Dimensions:   {width}×{height} px\n"
            f"• Megapixels:   {megapixels:.2f} MP\n"
            f"• Channels:     {channels}\n"
            f"• Transparency: {'Yes' if has_alpha else 'No'}\n"
            f"• File size:    {file_size / 1024:.1f} KB\n"
            f"• EXIF fields:  {exif_count}\n"
            f"• File path:    {filepath}"
        )

    except ImportError:
        return (
            "❌ Pillow is not installed.\n"
            "Run: python -m pip install Pillow"
        )

    except Exception:
        return (
            "❌ Image info error. "
            "The image metadata could not be read."
        )



# ============================================================
# ── RAG / PDF QUERY TOOL ───────────────────────────────────
# ============================================================

from langchain_core.runnables import RunnableConfig


@tool
def query_pdf(
    question: str,
    thread_id: str = "",
    top_k: int = 7,
    config: RunnableConfig = None,
) -> str:
    """
    Retrieve relevant content from PDFs uploaded in the current conversation.

    Uses the conversation's thread_id to access the correct PDF index and
    Maximal Marginal Relevance (MMR) retrieval to return relevant and
    diverse document chunks.

    IMPORTANT:
    Only use this tool when the user explicitly asks about an uploaded PDF,
    document, notes, or file.

    Do NOT use this tool for general knowledge questions.

    Args:
        question: Question to answer using the uploaded PDF.
        thread_id: Current conversation thread ID. Usually obtained
                   automatically from RunnableConfig.
        top_k: Number of chunks to retrieve (5-10, default 7).
        config: LangGraph runtime configuration containing thread_id.
    """

    # --------------------------------------------------------
    # Resolve thread_id from LangGraph runtime configuration
    # --------------------------------------------------------
    try:
        configurable = (config or {}).get("configurable", {})

        runtime_thread_id = configurable.get("thread_id")

        if runtime_thread_id:
            thread_id = runtime_thread_id

    except Exception:
        pass

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------
    if not isinstance(question, str):
        return "⚠️ PDF question must be text."

    question = question.strip()

    if not question:
        return "⚠️ Please provide a question about the uploaded PDF."

    if len(question) > 2000:
        return "⚠️ PDF question is too long. Please keep it under 2,000 characters."

    # --------------------------------------------------------
    # Validate thread ID
    # --------------------------------------------------------
    if not isinstance(thread_id, str) or not thread_id.strip():
        return (
            "⚠️ No conversation thread ID was available. "
            "The PDF index cannot be identified."
        )

    thread_id = thread_id.strip()

    # Prevent accidentally passing an enormous identifier.
    if len(thread_id) > 200:
        return "⚠️ Invalid conversation thread ID."

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return "⚠️ top_k must be an integer."

    top_k = max(5, min(top_k, 10))

    # --------------------------------------------------------
    # Get conversation-specific retriever
    # --------------------------------------------------------
    try:
        retriever = get_rag_retriever(
            thread_id,
            k=top_k
        )

    except Exception:
        return (
            "❌ The PDF search index could not be loaded. "
            "Please try again or re-upload the PDF."
        )

    if retriever is None:
        return (
            "📄 **No PDF indexed for this conversation.**\n\n"
            "Please upload a PDF using the "
            "'📄 Upload PDF' panel in the sidebar first, "
            "then ask your question again."
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------
    try:
        docs = retriever.invoke(question)

    except Exception:
        return (
            "❌ PDF retrieval failed while searching the document. "
            "Please try rephrasing your question."
        )

    if not docs:
        return (
            "🔍 **No relevant content found.**\n\n"
            "The uploaded PDF does not appear to contain enough "
            "relevant text for this question.\n\n"
            "Try rephrasing the question or asking about a topic "
            "that is explicitly covered in the document."
        )

    # --------------------------------------------------------
    # Format retrieved chunks
    # --------------------------------------------------------
    parts = []
    total_chars = 0

    # Keep the context reasonably bounded so that a large PDF
    # retrieval does not overwhelm the model.
    MAX_CONTEXT_CHARS = 30_000
    MAX_CHUNK_CHARS = 6_000

    for i, doc in enumerate(docs, 1):

        if not doc or not getattr(doc, "page_content", None):
            continue

        content = doc.page_content.strip()

        if not content:
            continue

        # Prevent one enormous chunk from dominating context.
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS].rstrip() + "\n[…chunk truncated…]"

        metadata = getattr(doc, "metadata", {}) or {}

        # Different PDF loaders use different metadata names.
        page = (
            metadata.get("page")
            or metadata.get("page_number")
            or metadata.get("page_num")
            or "?"
        )

        # Most PDF libraries use zero-based page indexes.
        # Keep the original value rather than silently changing it.
        source = (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
            or ""
        )

        if source:
            filename = os.path.basename(str(source))
        else:
            filename = "uploaded PDF"

        # Some vector stores attach chunk IDs.
        chunk_id = (
            metadata.get("chunk_id")
            or metadata.get("id")
            or ""
        )

        header = f"【Chunk {i} · {filename} · Page {page}"

        if chunk_id:
            header += f" · ID {chunk_id}"

        header += "】"

        part = f"{header}\n{content}"

        # Respect overall context limit.
        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        if len(part) > remaining:
            part = part[:remaining].rstrip() + "\n[…context truncated…]"

        parts.append(part)
        total_chars += len(part)

    if not parts:
        return (
            "🔍 The PDF index returned documents, but they contained "
            "no usable text."
        )

    # --------------------------------------------------------
    # Build model-facing retrieval context
    # --------------------------------------------------------
    context = "\n\n---\n\n".join(parts)

    return (
        f"📄 **PDF Retrieval Results**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Chunks retrieved: {len(parts)}\n"
        f"• Requested top_k: {top_k}\n"
        f"• Retrieval: MMR\n"
        f"• Context size: {total_chars:,} characters\n\n"
        f"{context}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Answer the user's question using ONLY the retrieved "
        f"PDF content above.**\n\n"
        f"User question: {question}\n\n"
        f"If the retrieved content does not contain enough information "
        f"to answer the question, explicitly say that the PDF does "
        f"not provide enough information rather than inventing an answer."
    )




# ============================================================
# Tool Registry
# ============================================================

tools = [
    # Original tools
    get_current_datetime,
    calculate,
    get_weather,
    search_wikipedia,
    search_web,
    get_news,
    fetch_webpage,
    convert_units,
    analyze_text,
    generate_uuid,
    random_number,
    # Knowledge & Research
    search_arxiv,
    search_pubmed,
    stackoverflow_search,
    # Code & Developer
    run_python,
    github_search,
    lint_code,
    # Finance & Crypto
    get_stock_price,
    get_crypto_price,
    get_forex_rate,
    portfolio_calculator,
    # File & Data
    read_csv,
    extract_pdf_text,
    read_docx,
    query_sql,
    # Location & Maps
    geocode_address,
    get_timezone,
    ip_geolocation,
    # AI Memory
    remember_fact,
    recall_memories,
    forget_memory,
    search_past_conversations,
    # Image
    resize_image,
    get_image_info,
    # RAG / PDF Q&A
    query_pdf,
]


# ============================================================
# LLM  (tools bound here so the model knows what is available)
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(tools)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are a powerful AI assistant with a large suite of tools. Use them proactively.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 DATETIME & UTILITIES
  get_current_datetime   — Current date/time for any timezone
  calculate              — Safe math expression evaluator
  convert_units          — Length, weight, volume, temp, speed, area
  analyze_text           — Word/sentence/paragraph statistics
  generate_uuid          — Generate unique UUIDs
  random_number          — Random integers or floats

🌦️ WEATHER & LOCATION
  get_weather            — Live weather for any city
  geocode_address        — Address → lat/lon coordinates
  get_timezone           — Timezone & local time for any city
  ip_geolocation         — Geolocate any IP address

🔍 SEARCH & WEB
  search_web             — Web search with multi-provider fallback (DDG → Bing)
  get_news               — Latest headlines via RSS feeds (Nepal, world, any topic)
  fetch_webpage          — Read any public URL
  search_wikipedia       — Wikipedia summaries

📚 KNOWLEDGE & RESEARCH
  search_arxiv           — Academic papers from arXiv
  search_pubmed          — Medical research via PubMed/NCBI
  stackoverflow_search   — Programming Q&A from Stack Overflow

💻 CODE & DEVELOPER
  run_python             — Execute Python code in a sandbox
  github_search          — Search GitHub repos, code, issues
  lint_code              — Check Python code for syntax errors

💹 FINANCE
  get_stock_price        — Live stock prices & metrics
  get_crypto_price       — Cryptocurrency prices (CoinGecko)
  get_forex_rate         — Foreign exchange rates
  portfolio_calculator   — Calculate total portfolio value

🗂️ FILE & DATA
  read_csv               — Preview and analyze CSV files
  extract_pdf_text       — Extract text from PDF files
  read_docx              — Read Word (.docx) documents
  query_sql              — Run SELECT queries on SQLite databases

🖼️ IMAGE
  resize_image           — Resize image files
  get_image_info         — Get image dimensions and metadata

🧠 AI MEMORY (persistent across sessions)
  remember_fact          — Store a fact for later recall
  recall_memories        — Retrieve stored memories
  forget_memory          — Delete a specific memory
  search_past_conversations — Search conversation history

📄 PDF / RAG (document question answering)
  query_pdf              — Answer questions from uploaded PDF using MMR semantic search

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Guidelines:
- Use tools proactively when a question calls for live data, calculations, or lookups.
- For ANY news request (country, topic, today's news), ALWAYS call get_news first.
  If get_news returns limited results, follow up with search_web for the same query.
  NEVER say you cannot access news — always try get_news then search_web.
- Combine multiple tool calls when needed (e.g., search web then fetch a URL).
- After receiving tool results, synthesize them into a clear, concise answer.
- If a tool returns an error, try a different tool — never give up after one failure.
- Use remember_fact when users ask you to remember something for future conversations.
- Always recall_memories at the start if the user mentions something you should know.
- For simple factual questions you already know, answer directly without using tools.

PDF / RAG RULES (very important):
- Call query_pdf ONLY when the user explicitly references the PDF / document / uploaded file / notes.
  Trigger phrases: "from the PDF", "from my document", "based on the file", "from my notes",
  "according to the uploaded document", "explain using the PDF", "what does the PDF say about X".
- Always pass the current thread_id when calling query_pdf (it is provided in the system context below).
- After receiving the retrieved chunks, synthesize a coherent answer — do NOT just dump raw chunks.
- If the PDF hasn't been uploaded yet, tell the user to upload it via the sidebar.
- You may call query_pdf multiple times with different phrasings if the first result is insufficient.

CURRENT THREAD ID: {thread_id}
"""


# ============================================================
# Agent Node
# ============================================================

from langchain_core.runnables import RunnableConfig

def agent_node(state: ChatState, config: RunnableConfig):
    """
    Main agent node. Reads thread_id from LangGraph's RunnableConfig so the
    LLM can pass it through to query_pdf.
    """
    # 1. Extract thread_id directly from the injected config parameter
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    # 2. Format the system prompt with the current thread_id
    system_prompt = SYSTEM_PROMPT.format(thread_id=thread_id)
    messages = state["messages"]
    full_messages = [SystemMessage(content=system_prompt)] + list(messages)
    
    # 3. Invoke the LLM (passing config here is also good practice for tracing)
    response = llm.invoke(full_messages, config=config)
    
    return {"messages": [response]}


# ============================================================
# Build Graph
# ============================================================

graph = StateGraph(ChatState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")


# ============================================================
# LangGraph Checkpointer
# ============================================================

checkpointer = SqliteSaver(conn)


# ============================================================
# Compile Chatbot
# ============================================================

chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================
# Conversation Functions
# ============================================================

def create_conversation(thread_id: str, title: str):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (thread_id, title) VALUES (?, ?)", (thread_id, title))
    conn.commit()


def update_conversation_title(thread_id: str, title: str):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
        (title, thread_id),
    )
    conn.commit()


def touch_conversation(thread_id: str):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
        (thread_id,),
    )
    conn.commit()


def delete_conversation(thread_id: str):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
    conn.commit()


def retrieve_all_threads():
    cursor = conn.cursor()
    cursor.execute(
        "SELECT thread_id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    )
    rows = cursor.fetchall()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]


def retrieve_thread_messages(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def conversation_exists(thread_id: str) -> bool:
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM conversations WHERE thread_id = ?", (thread_id,))
    return cursor.fetchone() is not None