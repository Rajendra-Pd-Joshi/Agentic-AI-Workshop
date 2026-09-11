import ast
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
        # Merge multiple FAISS stores into one
        vs = store_list[0]
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

@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """
    Get the current date and time for a given timezone.
    Examples of valid timezones: 'UTC', 'US/Eastern', 'Asia/Tokyo', 'Europe/London', 'Asia/Kathmandu'.
    Defaults to UTC if no timezone is provided.
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime(f"%A, %B %d, %Y — %I:%M:%S %p ({timezone})")
    except ZoneInfoNotFoundError:
        now = datetime.utcnow()
        return (
            f"Timezone '{timezone}' not recognized. "
            f"Falling back to UTC: {now.strftime('%A, %B %d, %Y — %I:%M:%S %p UTC')}"
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
        tree = ast.parse(expression.strip(), mode="eval")
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
        lat, lon = loc["latitude"], loc["longitude"]
        display_name = loc["name"]
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
        w = weather_resp.json().get("current", {})
        code = w.get("weather_code", -1)
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
        results = wikipedia.search(query, results=5)
        if not results:
            return f"No Wikipedia articles found for '{query}'."
        for candidate in results:
            try:
                page = wikipedia.page(candidate, auto_suggest=False)
                summary = wikipedia.summary(candidate, sentences=sentences, auto_suggest=False)
                return f"📖 **{page.title}**\n\n{summary}\n\n🔗 Source: {page.url}"
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue
        return f"Could not load a Wikipedia article for '{query}'."
    except ImportError:
        return "The 'wikipedia' package is not installed. Run: pip install wikipedia"
    except Exception as exc:
        return f"Wikipedia error: {exc}"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web and return top results. Tries multiple search backends
    (DuckDuckGo text, DuckDuckGo news, Bing scrape) so it always returns
    something useful even when one provider is rate-limited.
    Args:
        query: The search query (news topics, general questions, etc.).
        max_results: Number of results to return (default 5, max 10).
    """
    max_results = min(max_results, 10)

    # ── Strategy 1: DuckDuckGo text search ──────────────────
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"📌 **{r['title']}**\n   {r['body']}\n   🔗 {r['href']}"
                )
        if results:
            return f"🔍 Web search results for: *{query}*\n\n" + "\n\n".join(results)
    except Exception:
        pass

    # ── Strategy 2: DuckDuckGo news search ──────────────────
    try:
        from duckduckgo_search import DDGS
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
    except Exception:
        pass

    # ── Strategy 3: Bing scrape fallback ────────────────────
    try:
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max_results},
            headers=headers, timeout=10
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo")[:max_results]:
            title_el = item.select_one("h2 a")
            snippet_el = item.select_one(".b_caption p")
            if title_el:
                title = title_el.get_text(strip=True)
                link = title_el.get("href", "")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append(f"📌 **{title}**\n   {snippet}\n   🔗 {link}")
        if results:
            return f"🔍 Web results for: *{query}*\n\n" + "\n\n".join(results)
    except Exception:
        pass

    return (
        f"⚠️ All search providers are temporarily unavailable for '{query}'.\n"
        f"This is usually a rate-limit issue. Try again in a few seconds, "
        f"or use fetch_webpage with a direct URL."
    )


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

    max_results = min(max_results, 20)

    # Build list of feeds to try based on inputs
    combined = f"{topic} {country}".lower().strip()

    # Country/region-specific RSS feeds
    COUNTRY_FEEDS = {
        "nepal": [
            ("The Himalayan Times",  "https://thehimalayantimes.com/feed/"),
            ("Kathmandu Post",       "https://kathmandupost.com/rss"),
            ("OnlineKhabar English", "https://english.onlinekhabar.com/feed"),
            ("My Republica",         "https://myrepublica.nagariknetwork.com/rss/"),
            ("Rising Nepal Daily",   "https://risingnepaldaily.com/feed"),
            ("Nepal News",           "https://nepalnews.com/feed"),
            ("Setopati English",     "https://setopati.com/feed"),
        ],
        "india": [
            ("Times of India",   "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
            ("NDTV",             "https://feeds.feedburner.com/ndtvnews-top-stories"),
            ("The Hindu",        "https://www.thehindu.com/feeder/default.rss"),
        ],
        "us": [
            ("NPR",     "https://feeds.npr.org/1001/rss.xml"),
            ("CNN",     "http://rss.cnn.com/rss/edition.rss"),
            ("Reuters", "https://feeds.reuters.com/reuters/topNews"),
        ],
        "uk": [
            ("BBC",         "https://feeds.bbci.co.uk/news/rss.xml"),
            ("The Guardian", "https://www.theguardian.com/uk/rss"),
            ("Sky News",     "https://feeds.skynews.com/feeds/rss/world.xml"),
        ],
    }

    # General/international feeds
    GENERAL_FEEDS = [
        ("BBC World",       "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Reuters",         "https://feeds.reuters.com/reuters/topNews"),
        ("Al Jazeera",      "https://www.aljazeera.com/xml/rss/all.xml"),
        ("AP News",         "https://rsshub.app/apnews/topics/apf-topnews"),
        ("Google News",     f"https://news.google.com/rss/search?q={topic or 'world+news'}&hl=en&gl=US&ceid=US:en"),
        ("NPR",             "https://feeds.npr.org/1001/rss.xml"),
    ]

    # Select which feeds to query
    feeds_to_try = []
    for key, feed_list in COUNTRY_FEEDS.items():
        if key in combined:
            feeds_to_try = feed_list + GENERAL_FEEDS[:2]
            break
    if not feeds_to_try:
        feeds_to_try = GENERAL_FEEDS

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    all_entries = []
    sources_used = []

    for feed_name, feed_url in feeds_to_try:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=8)
            if resp.status_code != 200:
                continue
            feed = feedparser.parse(resp.content)
            if not feed.entries:
                continue
            sources_used.append(feed_name)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link  = entry.get("link", "")
                summary = entry.get("summary", entry.get("description", "")).strip()
                # Strip HTML tags from summary
                summary = re.sub(r"<[^>]+>", "", summary)[:200]
                # Parse published date
                pub = entry.get("published", entry.get("updated", ""))
                try:
                    import email.utils
                    pub_dt = email.utils.parsedate_to_datetime(pub)
                    pub_str = pub_dt.strftime("%b %d, %Y %H:%M")
                except Exception:
                    pub_str = pub[:16] if pub else ""
                # Filter by topic keyword if given
                if topic:
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

    if not all_entries:
        # Last resort: try DuckDuckGo news
        try:
            from duckduckgo_search import DDGS
            search_q = f"{topic} {country} news".strip()
            results = []
            with DDGS() as ddgs:
                for r in ddgs.news(search_q, max_results=max_results):
                    date = r.get("date", "")[:10]
                    results.append(
                        f"📰 **{r['title']}** ({date})\n"
                        f"   {r.get('body', '')[:200]}\n"
                        f"   Source: {r.get('source', '?')} · 🔗 {r['url']}"
                    )
            if results:
                label = f"{topic} {country}".strip() or "World"
                return f"📰 Latest news — *{label}*\n\n" + "\n\n".join(results[:max_results])
        except Exception:
            pass
        return (
            f"⚠️ Could not fetch news for '{topic or country or 'world'}' right now.\n"
            f"All RSS sources returned 403 (blocked by server) and DuckDuckGo is rate-limited.\n"
            f"Try using fetch_webpage with a direct URL like https://kathmandupost.com or https://thehimalayantimes.com"
        )

    # Deduplicate by title
    seen = set()
    unique_entries = []
    for e in all_entries:
        key = e["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)

    # Take top N
    top = unique_entries[:max_results]
    lines = []
    for e in top:
        pub_str = f" · {e['pub']}" if e["pub"] else ""
        summary_str = f"\n   {e['summary']}" if e["summary"] else ""
        lines.append(
            f"📰 **{e['title']}**{summary_str}\n"
            f"   {e['source']}{pub_str} · 🔗 {e['link']}"
        )

    label = " | ".join(dict.fromkeys(sources_used))
    header_topic = f"{topic} {country}".strip() or "World"
    return (
        f"📰 **Latest News — {header_topic}** ({len(top)} headlines)\n"
        f"Sources: {label}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(lines)
    )


@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and extract the readable text content from any public webpage URL.
    Useful for reading articles, documentation, or any page the user shares.
    Args:
        url: The full URL including https:// or http://
    """
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose()
        main = (
            soup.find("article") or soup.find("main")
            or soup.find(id="content") or soup.find(class_="content") or soup.body
        )
        text = (main or soup).get_text(separator="\n", strip=True)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)
        if len(text) > 4000:
            text = text[:4000] + "\n\n… [Content truncated to 4000 characters]"
        return f"📄 Content from {url}:\n\n{text}"
    except requests.HTTPError as exc:
        return f"HTTP error fetching page: {exc}"
    except requests.Timeout:
        return "Request timed out. The page may be slow or unavailable."
    except ImportError:
        return "The 'beautifulsoup4' package is not installed. Run: pip install beautifulsoup4"
    except Exception as exc:
        return f"Webpage fetch error: {exc}"


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a numeric value between common units of measurement.
    Supported categories:
      - Length:      km, miles, meters, feet, inches, cm, mm, yards
      - Weight/Mass: kg, lbs, grams, oz, mg, tonnes
      - Volume:      liters, ml, gallons, cups, pints, quarts, fl_oz
      - Temperature: celsius, fahrenheit, kelvin
      - Speed:       kmh, mph, ms (meters per second), knots
      - Area:        sqm, sqft, sqkm, sqmiles, acres, hectares
    """
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()
    temp_conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"):     lambda v: v + 273.15,
        ("kelvin", "celsius"):     lambda v: v - 273.15,
        ("fahrenheit", "kelvin"):  lambda v: (v - 32) * 5 / 9 + 273.15,
        ("kelvin", "fahrenheit"):  lambda v: (v - 273.15) * 9 / 5 + 32,
    }
    if (f, t) in temp_conversions:
        result = temp_conversions[(f, t)](value)
        unit_symbols = {"celsius": "°C", "fahrenheit": "°F", "kelvin": "K"}
        fs, ts = unit_symbols.get(f, f), unit_symbols.get(t, t)
        return f"{value}{fs} = {round(result, 6)}{ts}"
    if f == t:
        return f"{value} {f} = {value} {t}"
    length = {"meters": 1, "m": 1, "km": 1000, "kilometers": 1000, "miles": 1609.344, "mile": 1609.344, "feet": 0.3048, "ft": 0.3048, "inches": 0.0254, "inch": 0.0254, "in": 0.0254, "cm": 0.01, "centimeters": 0.01, "mm": 0.001, "millimeters": 0.001, "yards": 0.9144, "yd": 0.9144, "nautical_miles": 1852}
    weight = {"grams": 1, "g": 1, "kg": 1000, "kilograms": 1000, "lbs": 453.592, "pounds": 453.592, "lb": 453.592, "oz": 28.3495, "ounces": 28.3495, "mg": 0.001, "milligrams": 0.001, "tonnes": 1_000_000}
    volume = {"ml": 1, "milliliters": 1, "liters": 1000, "l": 1000, "gallons": 3785.41, "gal": 3785.41, "cups": 236.588, "pints": 473.176, "quarts": 946.353, "fl_oz": 29.5735, "tbsp": 14.7868, "tsp": 4.92892}
    speed  = {"ms": 1, "m/s": 1, "kmh": 0.277778, "km/h": 0.277778, "mph": 0.44704, "knots": 0.514444, "fps": 0.3048}
    area   = {"sqm": 1, "m2": 1, "sqft": 0.092903, "sqkm": 1_000_000, "sqmiles": 2_589_988.11, "acres": 4046.86, "hectares": 10_000, "ha": 10_000}
    for _, table in [("Length", length), ("Weight", weight), ("Volume", volume), ("Speed", speed), ("Area", area)]:
        if f in table and t in table:
            result = value * table[f] / table[t]
            return f"{value} {from_unit} = {round(result, 6)} {to_unit}"
    return f"Conversion from '{from_unit}' to '{to_unit}' is not supported.\nSupported: Length, Weight, Volume, Temperature, Speed, Area."


@tool
def analyze_text(text: str) -> str:
    """
    Analyze a block of text and return detailed statistics.
    Returns: character count, word count, unique words, sentence count,
             paragraph count, average word length, most common words.
    """
    words_raw = text.split()
    words_clean = [re.sub(r"[^\w']", "", w).lower() for w in words_raw if re.sub(r"[^\w']", "", w)]
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    unique_words = set(words_clean)
    avg_word_len = round(sum(len(w) for w in words_clean) / len(words_clean), 2) if words_clean else 0
    stopwords = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","was","are","were","it","its","this","that","i","you","he","she","we","they","be","been","have","has","had","do","does","did","will","would","could","should","may","might","not","from","by","as","so","if","my","your","our","their","his","her"}
    content_words = [w for w in words_clean if w not in stopwords and len(w) > 2]
    top_words = Counter(content_words).most_common(5)
    top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top_words) if top_words else "N/A"
    return (
        f"📊 Text Analysis\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Total characters:       {len(text)}\n"
        f"• Characters (no spaces): {len(text.replace(' ', ''))}\n"
        f"• Total words:            {len(words_raw)}\n"
        f"• Unique words:           {len(unique_words)}\n"
        f"• Sentences:              {len(sentences)}\n"
        f"• Paragraphs:             {len(paragraphs)}\n"
        f"• Avg word length:        {avg_word_len} chars\n"
        f"• Avg words/sentence:     {round(len(words_raw) / max(len(sentences), 1), 1)}\n"
        f"• Top content words:      {top_str}"
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
        max_results = min(max_results, 10)
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            abstract = paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary
            results.append(
                f"📄 **{paper.title}**\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   Abstract: {abstract}\n"
                f"   🔗 {paper.entry_id}"
            )
        if not results:
            return f"No arXiv papers found for '{query}'."
        return f"📚 arXiv results for: *{query}*\n\n" + "\n\n".join(results)
    except ImportError:
        return "The 'arxiv' package is not installed. Run: pip install arxiv"
    except Exception as exc:
        return f"arXiv search error: {exc}"


@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for medical and biomedical research papers via NCBI Entrez API.
    Args:
        query: Medical or biological research topic.
        max_results: Number of results to return (default 5, max 10).
    """
    try:
        max_results = min(max_results, 10)
        # Step 1: search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_resp = requests.get(search_url, params={
            "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"
        }, timeout=10)
        search_data = search_resp.json()
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed articles found for '{query}'."
        # Step 2: fetch summaries
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_resp = requests.get(summary_url, params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "json"
        }, timeout=10)
        summary_data = summary_resp.json().get("result", {})
        results = []
        for uid in ids:
            article = summary_data.get(uid, {})
            title = article.get("title", "Unknown title")
            authors = article.get("authors", [])
            author_names = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_names += " et al."
            pub_date = article.get("pubdate", "Unknown date")
            results.append(
                f"🔬 **{title}**\n"
                f"   Authors: {author_names}\n"
                f"   Published: {pub_date}\n"
                f"   🔗 https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            )
        return f"🏥 PubMed results for: *{query}*\n\n" + "\n\n".join(results)
    except Exception as exc:
        return f"PubMed search error: {exc}"


@tool
def stackoverflow_search(query: str, max_results: int = 5) -> str:
    """
    Search Stack Overflow for programming questions and answers.
    Args:
        query: Programming question or error message to search.
        max_results: Number of results to return (default 5).
    """
    try:
        url = "https://api.stackexchange.com/2.3/search/advanced"
        resp = requests.get(url, params={
            "order": "desc", "sort": "relevance", "q": query,
            "site": "stackoverflow", "pagesize": min(max_results, 10),
            "filter": "withbody"
        }, timeout=10)
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return f"No Stack Overflow results found for '{query}'."
        results = []
        for item in items:
            title = item.get("title", "Unknown")
            link = item.get("link", "")
            score = item.get("score", 0)
            answered = "✅ Answered" if item.get("is_answered") else "❓ Unanswered"
            answer_count = item.get("answer_count", 0)
            results.append(
                f"💻 **{title}**\n"
                f"   {answered} · {answer_count} answers · Score: {score}\n"
                f"   🔗 {link}"
            )
        return f"🔍 Stack Overflow results for: *{query}*\n\n" + "\n\n".join(results)
    except Exception as exc:
        return f"Stack Overflow search error: {exc}"


# ============================================================
# ── NEW: CODE & DEVELOPER TOOLS ─────────────────────────────
# ============================================================

@tool
def run_python(code: str) -> str:
    """
    Execute Python code in a sandboxed subprocess and return the output.
    Use this for data processing, calculations, generating outputs, etc.
    Args:
        code: Valid Python code to execute. Use print() to show results.
    Warning: Code runs in an isolated subprocess with a 10-second timeout.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True, timeout=10
        )
        os.unlink(tmp_path)
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


@tool
def github_search(query: str, search_type: str = "repositories", max_results: int = 5) -> str:
    """
    Search GitHub for repositories, code, or issues.
    Args:
        query: Search terms.
        search_type: One of 'repositories', 'code', 'issues', 'users' (default: repositories).
        max_results: Number of results to return (default 5, max 10).
    """
    try:
        valid_types = {"repositories", "code", "issues", "users"}
        if search_type not in valid_types:
            search_type = "repositories"
        url = f"https://api.github.com/search/{search_type}"
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(url, headers=headers, params={"q": query, "per_page": min(max_results, 10)}, timeout=10)
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return f"No GitHub {search_type} found for '{query}'."
        results = []
        for item in items:
            if search_type == "repositories":
                results.append(
                    f"⭐ **{item.get('full_name')}** ({item.get('stargazers_count', 0)} stars)\n"
                    f"   {item.get('description', 'No description')}\n"
                    f"   Language: {item.get('language', 'Unknown')} | Forks: {item.get('forks_count', 0)}\n"
                    f"   🔗 {item.get('html_url')}"
                )
            elif search_type == "issues":
                results.append(
                    f"🐛 **{item.get('title')}**\n"
                    f"   State: {item.get('state')} | Comments: {item.get('comments', 0)}\n"
                    f"   🔗 {item.get('html_url')}"
                )
            else:
                results.append(
                    f"• **{item.get('full_name') or item.get('login') or item.get('name')}**\n"
                    f"   🔗 {item.get('html_url')}"
                )
        total = data.get("total_count", len(items))
        return f"🐙 GitHub {search_type} for: *{query}* ({total:,} total)\n\n" + "\n\n".join(results)
    except Exception as exc:
        return f"GitHub search error: {exc}"


@tool
def lint_code(code: str, language: str = "python") -> str:
    """
    Lint Python code and return style issues and errors.
    Args:
        code: The source code to lint.
        language: Programming language (currently supports 'python').
    """
    if language.lower() != "python":
        return f"Linting for '{language}' is not yet supported. Only Python is supported."
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            ["python3", "-m", "py_compile", tmp_path],
            capture_output=True, text=True, timeout=10
        )
        syntax_errors = result.stderr.strip()
        os.unlink(tmp_path)
        if syntax_errors:
            return f"❌ Syntax errors found:\n```\n{syntax_errors}\n```"
        # Also check with ast
        try:
            ast.parse(code)
            return "✅ No syntax errors found. Code looks valid."
        except SyntaxError as e:
            return f"❌ Syntax error at line {e.lineno}: {e.msg}\n```\n{e.text}\n```"
    except Exception as exc:
        return f"Lint error: {exc}"


# ============================================================
# ── NEW: FINANCE & CRYPTO TOOLS ─────────────────────────────
# ============================================================

@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price and key metrics for any publicly traded company.
    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL', 'GOOGL', 'TSLA', 'MSFT').
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        hist = ticker.history(period="2d")
        if hist.empty:
            return f"No data found for ticker '{symbol}'. Please check the symbol."
        current = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[0] if len(hist) > 1 else current
        change = current - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        direction = "📈" if change >= 0 else "📉"
        return (
            f"{direction} **{info.get('shortName', symbol.upper())} ({symbol.upper()})**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Current Price:  ${current:.2f}\n"
            f"• Change:         {'+' if change >= 0 else ''}{change:.2f} ({change_pct:+.2f}%)\n"
            f"• Market Cap:     ${info.get('marketCap', 0):,.0f}\n"
            f"• 52-Week High:   ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"• 52-Week Low:    ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"• P/E Ratio:      {info.get('trailingPE', 'N/A')}\n"
            f"• Volume:         {info.get('volume', 'N/A'):,}"
        )
    except ImportError:
        return "The 'yfinance' package is not installed. Run: pip install yfinance"
    except Exception as exc:
        return f"Stock price error: {exc}"


@tool
def get_crypto_price(coin_id: str) -> str:
    """
    Get the current price and market data for any cryptocurrency via CoinGecko API (free, no key needed).
    Args:
        coin_id: CoinGecko coin ID (e.g. 'bitcoin', 'ethereum', 'solana', 'dogecoin').
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        resp = requests.get(url, params={
            "vs_currency": "usd",
            "ids": coin_id.lower(),
            "price_change_percentage": "24h"
        }, timeout=10)
        data = resp.json()
        if not data:
            return f"No data found for crypto '{coin_id}'. Check the coin ID (e.g. 'bitcoin', 'ethereum')."
        c = data[0]
        change_24h = c.get("price_change_percentage_24h", 0) or 0
        direction = "📈" if change_24h >= 0 else "📉"
        return (
            f"{direction} **{c.get('name')} ({c.get('symbol', '').upper()})**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Current Price:   ${c.get('current_price', 0):,.4f}\n"
            f"• 24h Change:      {change_24h:+.2f}%\n"
            f"• Market Cap:      ${c.get('market_cap', 0):,.0f}\n"
            f"• 24h Volume:      ${c.get('total_volume', 0):,.0f}\n"
            f"• 24h High:        ${c.get('high_24h', 0):,.4f}\n"
            f"• 24h Low:         ${c.get('low_24h', 0):,.4f}\n"
            f"• All-Time High:   ${c.get('ath', 0):,.4f}\n"
            f"• Rank:            #{c.get('market_cap_rank', 'N/A')}"
        )
    except Exception as exc:
        return f"Crypto price error: {exc}"


@tool
def get_forex_rate(from_currency: str, to_currency: str) -> str:
    """
    Get the current foreign exchange rate between two currencies.
    Uses yfinance — free and no API key required.
    Args:
        from_currency: Source currency code (e.g. 'USD', 'EUR', 'NPR', 'GBP').
        to_currency: Target currency code (e.g. 'EUR', 'JPY', 'INR', 'AUD').
    """
    try:
        import yfinance as yf
        pair = f"{from_currency.upper()}{to_currency.upper()}=X"
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="2d")
        if hist.empty:
            return f"Could not fetch exchange rate for {from_currency}/{to_currency}."
        rate = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[0] if len(hist) > 1 else rate
        change = rate - prev
        change_pct = (change / prev) * 100 if prev else 0
        return (
            f"💱 **{from_currency.upper()} → {to_currency.upper()}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Rate:    1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}\n"
            f"• Change:  {change:+.4f} ({change_pct:+.2f}% today)"
        )
    except ImportError:
        return "The 'yfinance' package is not installed. Run: pip install yfinance"
    except Exception as exc:
        return f"Forex rate error: {exc}"


@tool
def portfolio_calculator(holdings: str) -> str:
    """
    Calculate total portfolio value given a list of stock/crypto holdings.
    Args:
        holdings: Comma-separated list in format 'SYMBOL:QUANTITY' 
                  e.g. 'AAPL:10,GOOGL:5,BTC-USD:0.5'
                  Use standard ticker symbols. For crypto use yfinance format like BTC-USD.
    """
    try:
        import yfinance as yf
        items = [h.strip() for h in holdings.split(",")]
        rows = []
        total = 0.0
        for item in items:
            if ":" not in item:
                continue
            symbol, qty_str = item.split(":", 1)
            symbol = symbol.strip().upper()
            qty = float(qty_str.strip())
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if hist.empty:
                    rows.append(f"• {symbol}: ❌ No data found")
                    continue
                price = hist["Close"].iloc[-1]
                value = price * qty
                total += value
                rows.append(f"• {symbol}: {qty} × ${price:,.2f} = **${value:,.2f}**")
            except Exception:
                rows.append(f"• {symbol}: ❌ Could not fetch price")
        if not rows:
            return "No valid holdings found. Format: 'AAPL:10,GOOGL:5'"
        return (
            f"💼 **Portfolio Summary**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(rows)
            + f"\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 **Total Value: ${total:,.2f}**"
        )
    except ImportError:
        return "The 'yfinance' package is not installed. Run: pip install yfinance"
    except Exception as exc:
        return f"Portfolio error: {exc}"


# ============================================================
# ── NEW: FILE & DATA TOOLS ──────────────────────────────────
# ============================================================

@tool
def read_csv(filepath: str, max_rows: int = 20) -> str:
    """
    Read and preview a CSV file, showing its structure and first rows.
    Args:
        filepath: Path to the CSV file on disk.
        max_rows: Number of rows to preview (default 20).
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        shape = df.shape
        dtypes = df.dtypes.to_string()
        preview = df.head(max_rows).to_string(index=False)
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0]
        null_info = null_cols.to_string() if not null_cols.empty else "None"
        return (
            f"📊 **CSV File: {filepath}**\n"
            f"Shape: {shape[0]} rows × {shape[1]} columns\n\n"
            f"**Column Types:**\n{dtypes}\n\n"
            f"**Missing Values:**\n{null_info}\n\n"
            f"**Preview (first {min(max_rows, shape[0])} rows):**\n```\n{preview}\n```"
        )
    except ImportError:
        return "The 'pandas' package is not installed. Run: pip install pandas"
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as exc:
        return f"CSV read error: {exc}"


@tool
def extract_pdf_text(filepath: str, max_pages: int = 5) -> str:
    """
    Extract and return text content from a PDF file.
    Args:
        filepath: Path to the PDF file.
        max_pages: Maximum number of pages to extract (default 5).
    """
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(max_pages, total_pages)
            text_parts = []
            for i, page in enumerate(pdf.pages[:pages_to_read]):
                text = page.extract_text() or ""
                text_parts.append(f"--- Page {i+1} ---\n{text}")
            full_text = "\n\n".join(text_parts)
            if len(full_text) > 5000:
                full_text = full_text[:5000] + "\n\n… [Truncated to 5000 chars]"
        return (
            f"📄 **PDF: {filepath}** ({total_pages} pages total)\n"
            f"Showing {pages_to_read} page(s):\n\n{full_text}"
        )
    except ImportError:
        return "The 'pdfplumber' package is not installed. Run: pip install pdfplumber"
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as exc:
        return f"PDF extract error: {exc}"


@tool
def read_docx(filepath: str) -> str:
    """
    Read and extract text content from a Word (.docx) document.
    Args:
        filepath: Path to the .docx file.
    """
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        if len(text) > 5000:
            text = text[:5000] + "\n\n… [Truncated to 5000 chars]"
        return (
            f"📝 **Word Document: {filepath}**\n"
            f"({len(paragraphs)} paragraphs)\n\n{text}"
        )
    except ImportError:
        return "The 'python-docx' package is not installed. Run: pip install python-docx"
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as exc:
        return f"DOCX read error: {exc}"


@tool
def query_sql(database_path: str, sql_query: str) -> str:
    """
    Execute a SQL SELECT query on a SQLite database file and return results.
    Only SELECT queries are allowed for safety.
    Args:
        database_path: Path to the SQLite .db file.
        sql_query: A valid SQL SELECT query.
    """
    try:
        if not sql_query.strip().upper().startswith("SELECT"):
            return "❌ Only SELECT queries are allowed for safety."
        db_conn = sqlite3.connect(database_path, check_same_thread=False)
        cursor = db_conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        db_conn.close()
        if not rows:
            return "Query returned no results."
        header = " | ".join(columns)
        separator = "-" * len(header)
        result_rows = [" | ".join(str(v) for v in row) for row in rows[:50]]
        body = "\n".join(result_rows)
        note = f"\n(Showing {len(rows)} rows)" if len(rows) <= 50 else f"\n(Showing first 50 of {len(rows)} rows)"
        return f"🗄️ **SQL Query Result**\n```\n{header}\n{separator}\n{body}\n```{note}"
    except FileNotFoundError:
        return f"Database not found: {database_path}"
    except sqlite3.Error as exc:
        return f"SQL error: {exc}"
    except Exception as exc:
        return f"Query error: {exc}"


# ============================================================
# ── NEW: LOCATION & MAPS TOOLS ──────────────────────────────
# ============================================================

@tool
def geocode_address(address: str) -> str:
    """
    Geocode an address or place name to get latitude, longitude, and details.
    Uses Open-Meteo Geocoding API — free, no key needed.
    Args:
        address: Any address, city name, or place (e.g. 'Eiffel Tower', 'Sydney, Australia').
    """
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": address, "count": 3, "language": "en", "format": "json"},
            timeout=10
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return f"Could not geocode '{address}'. Try a more specific location."
        output = [f"📍 Geocoding results for: *{address}*\n"]
        for i, loc in enumerate(results[:3], 1):
            output.append(
                f"{i}. **{loc.get('name')}, {loc.get('country', '')}**\n"
                f"   Latitude:  {loc.get('latitude')}\n"
                f"   Longitude: {loc.get('longitude')}\n"
                f"   Timezone:  {loc.get('timezone', 'N/A')}\n"
                f"   Admin:     {loc.get('admin1', 'N/A')}"
            )
        return "\n\n".join(output)
    except Exception as exc:
        return f"Geocoding error: {exc}"


@tool
def get_timezone(location: str) -> str:
    """
    Get the timezone and current local time for any city or location.
    Args:
        location: City or place name (e.g. 'Tokyo', 'New York', 'Kathmandu').
    """
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return f"Could not find timezone for '{location}'."
        loc = results[0]
        tz_name = loc.get("timezone", "UTC")
        try:
            tz = ZoneInfo(tz_name)
            local_time = datetime.now(tz).strftime("%A, %B %d, %Y — %I:%M:%S %p")
        except Exception:
            local_time = "Could not compute local time"
        return (
            f"🕐 **Timezone for {loc.get('name')}, {loc.get('country', '')}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Timezone:    {tz_name}\n"
            f"• Local Time:  {local_time}\n"
            f"• Coordinates: {loc.get('latitude')}, {loc.get('longitude')}"
        )
    except Exception as exc:
        return f"Timezone lookup error: {exc}"


@tool
def ip_geolocation(ip_address: str = "") -> str:
    """
    Get geolocation information for an IP address.
    Leave ip_address blank to geolocate the current machine's public IP.
    Args:
        ip_address: IPv4 or IPv6 address (optional — defaults to current public IP).
    """
    try:
        target = ip_address.strip() if ip_address.strip() else ""
        url = f"http://ip-api.com/json/{target}" if target else "http://ip-api.com/json/"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") == "fail":
            return f"Failed to geolocate IP '{ip_address}': {data.get('message')}"
        return (
            f"🌐 **IP Geolocation: {data.get('query')}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Country:   {data.get('country')} ({data.get('countryCode')})\n"
            f"• Region:    {data.get('regionName')}\n"
            f"• City:      {data.get('city')}\n"
            f"• ZIP:       {data.get('zip', 'N/A')}\n"
            f"• Latitude:  {data.get('lat')}\n"
            f"• Longitude: {data.get('lon')}\n"
            f"• Timezone:  {data.get('timezone')}\n"
            f"• ISP:       {data.get('isp')}"
        )
    except Exception as exc:
        return f"IP geolocation error: {exc}"


# ============================================================
# ── NEW: AI MEMORY TOOLS ────────────────────────────────────
# ============================================================

@tool
def remember_fact(key: str, value: str) -> str:
    """
    Store a fact or piece of information in persistent memory so it can be recalled later.
    Use this when the user tells you something they want you to remember across conversations.
    Args:
        key: A short descriptive label (e.g. 'user_name', 'favorite_color', 'project_goal').
        value: The information to store.
    """
    try:
        cursor = conn.cursor()
        # Upsert: delete old value for key, then insert new one
        cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
        cursor.execute("INSERT INTO memories (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        return f"✅ Remembered: **{key}** → \"{value}\""
    except Exception as exc:
        return f"Memory store error: {exc}"


@tool
def recall_memories(query: str = "") -> str:
    """
    Recall stored memories. Optionally filter by a keyword.
    Args:
        query: Optional keyword to filter memories (leave blank to retrieve all).
    """
    try:
        cursor = conn.cursor()
        if query.strip():
            cursor.execute(
                "SELECT key, value, created_at FROM memories WHERE key LIKE ? OR value LIKE ? ORDER BY created_at DESC",
                (f"%{query}%", f"%{query}%")
            )
        else:
            cursor.execute("SELECT key, value, created_at FROM memories ORDER BY created_at DESC")
        rows = cursor.fetchall()
        if not rows:
            return "🧠 No memories stored yet." if not query else f"No memories found matching '{query}'."
        lines = [f"🧠 **Stored Memories** ({len(rows)} total):\n"]
        for key, value, created_at in rows:
            lines.append(f"• **{key}**: {value}  *(saved {created_at[:10]})*")
        return "\n".join(lines)
    except Exception as exc:
        return f"Memory recall error: {exc}"


@tool
def forget_memory(key: str) -> str:
    """
    Delete a specific stored memory by its key.
    Args:
        key: The key of the memory to delete.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
        conn.commit()
        if cursor.rowcount > 0:
            return f"🗑️ Deleted memory: **{key}**"
        return f"No memory found with key '{key}'."
    except Exception as exc:
        return f"Memory delete error: {exc}"


@tool
def search_past_conversations(keyword: str, max_results: int = 5) -> str:
    """
    Search through past conversation titles for a keyword.
    Args:
        keyword: Word or phrase to search for in conversation titles.
        max_results: Maximum number of results to return (default 5).
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, title, updated_at FROM conversations WHERE title LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{keyword}%", max_results)
        )
        rows = cursor.fetchall()
        if not rows:
            return f"No past conversations found matching '{keyword}'."
        lines = [f"🔍 **Conversations matching '{keyword}':**\n"]
        for thread_id, title, updated_at in rows:
            lines.append(f"• **{title}** (last active: {updated_at[:10]})")
        return "\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"


# ============================================================
# ── NEW: IMAGE TOOLS ────────────────────────────────────────
# ============================================================

@tool
def resize_image(filepath: str, width: int, height: int, output_path: str = "") -> str:
    """
    Resize an image file to the specified dimensions.
    Args:
        filepath: Path to the source image (JPG, PNG, BMP, etc.).
        width: Target width in pixels.
        height: Target height in pixels.
        output_path: Where to save the resized image (optional — defaults to same dir with '_resized' suffix).
    """
    try:
        from PIL import Image
        img = Image.open(filepath)
        original_size = img.size
        img_resized = img.resize((width, height), Image.LANCZOS)
        if not output_path:
            base, ext = os.path.splitext(filepath)
            output_path = f"{base}_resized{ext}"
        img_resized.save(output_path)
        return (
            f"✅ Image resized successfully!\n"
            f"• Original size: {original_size[0]}×{original_size[1]} px\n"
            f"• New size:      {width}×{height} px\n"
            f"• Saved to:      {output_path}"
        )
    except ImportError:
        return "The 'Pillow' package is not installed. Run: pip install Pillow"
    except FileNotFoundError:
        return f"Image file not found: {filepath}"
    except Exception as exc:
        return f"Image resize error: {exc}"


@tool
def get_image_info(filepath: str) -> str:
    """
    Get metadata and information about an image file (dimensions, format, mode, file size).
    Args:
        filepath: Path to the image file.
    """
    try:
        from PIL import Image
        img = Image.open(filepath)
        file_size = os.path.getsize(filepath)
        return (
            f"🖼️ **Image Info: {os.path.basename(filepath)}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Format:     {img.format}\n"
            f"• Mode:       {img.mode}\n"
            f"• Dimensions: {img.size[0]}×{img.size[1]} px\n"
            f"• File Size:  {file_size / 1024:.1f} KB\n"
            f"• File Path:  {filepath}"
        )
    except ImportError:
        return "The 'Pillow' package is not installed. Run: pip install Pillow"
    except FileNotFoundError:
        return f"Image file not found: {filepath}"
    except Exception as exc:
        return f"Image info error: {exc}"


# ============================================================
# ── RAG TOOL ────────────────────────────────────────────────
# ============================================================

@tool
def query_pdf(question: str, thread_id: str, top_k: int = 7) -> str:
    """
    Answer a question using the PDF documents the user has uploaded in this conversation.
    Uses semantic search with Maximal Marginal Relevance (MMR) to retrieve the most
    relevant AND diverse chunks from the uploaded PDF(s), then returns them as context.

    IMPORTANT: Only call this tool when the user explicitly mentions the PDF, uploaded
    document, or asks something like 'from the PDF', 'based on the document', 'according
    to the file', 'from my notes', 'using the uploaded file', etc.
    Do NOT call this for general questions — use other tools or your own knowledge instead.

    Args:
        question: The user's question to answer from the PDF content.
        thread_id: The current conversation thread ID (passed automatically by the agent).
        top_k: Number of document chunks to retrieve (5–10; default 7).
    """
    top_k = max(5, min(top_k, 10))
    retriever = get_rag_retriever(thread_id, k=top_k)

    if retriever is None:
        return (
            "📄 **No PDF indexed for this conversation.**\n"
            "Please upload a PDF using the '📄 Upload PDF' panel in the sidebar first, "
            "then ask your question again."
        )

    try:
        docs = retriever.invoke(question)
    except Exception as exc:
        return f"RAG retrieval error: {exc}"

    if not docs:
        return (
            "🔍 No relevant content found in the uploaded PDF for your question.\n"
            "Try rephrasing or asking about a different topic covered in the document."
        )

    # Format retrieved chunks
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        page = meta.get("page", meta.get("page_number", "?"))
        source = meta.get("source", "")
        filename = os.path.basename(source) if source else "uploaded PDF"
        parts.append(
            f"【Chunk {i} · {filename} · Page {page}】\n{doc.page_content.strip()}"
        )

    context = "\n\n---\n\n".join(parts)
    return (
        f"📄 **Retrieved {len(docs)} chunks from the uploaded PDF** "
        f"(MMR, top_k={top_k}):\n\n"
        f"{context}\n\n"
        f"---\n"
        f"*Use the content above to answer the user's question: \"{question}\"*"
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

def agent_node(state: ChatState, config: dict):
    """
    Main agent node. Injects the current thread_id into the system prompt so
    the LLM can pass it through to query_pdf.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    system_prompt = SYSTEM_PROMPT.format(thread_id=thread_id)
    messages = state["messages"]
    full_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm.invoke(full_messages)
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