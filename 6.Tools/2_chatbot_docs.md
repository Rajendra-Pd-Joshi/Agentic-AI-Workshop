# 🤖 LangGraph Chatbot

A powerful, multi-tool AI chatbot built with **LangGraph**, **LangChain**, and **Streamlit** — featuring persistent conversations, live tool execution, and AI memory across sessions.

---

## 🏗️ Architecture

```
User Input (Streamlit UI)
        │
        ▼
  ┌─────────────┐
  │  Agent Node │  ◄── SystemPrompt + Message History
  │  (GPT-4o)   │
  └──────┬──────┘
         │ tool_calls?
    ┌────┴────┐
   YES       NO
    │         │
    ▼         ▼
┌────────┐  END (stream
│ Tools  │   to user)
│  Node  │
└────┬───┘
     │ results
     └──► Agent Node (loop)
```

The graph loops between the **agent** and **tools** nodes until the LLM produces a final text response, at which point it streams back to the user.

---

## 🛠️ Tech Stack

| Layer                  | Technology                  |
| ---------------------- | --------------------------- |
| UI                     | Streamlit                   |
| Orchestration          | LangGraph                   |
| LLM                    | GPT-4o-mini (OpenAI)        |
| Memory / Checkpointing | SQLite via `SqliteSaver`    |
| Tool Framework         | LangChain `@tool` decorator |
| Conversation Store     | SQLite (`chatbot.db`)       |

---

## 📁 Project Structure

```
.
├── app.py          # Streamlit frontend — UI, streaming, sidebar
├── backend.py      # LangGraph graph, all tools, DB functions
├── chatbot.db      # SQLite database (auto-created on first run)
├── .env            # API keys (OPENAI_API_KEY, etc.)
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone / copy the files

```bash
mkdir my-chatbot && cd my-chatbot
# place app.py and backend.py here
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit langgraph langchain langchain-openai langchain-core \
            openai python-dotenv requests \
            arxiv yfinance pandas openpyxl pdfplumber python-docx Pillow \
            duckduckgo-search wikipedia beautifulsoup4 chromadb
```

### 4. Set your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...        # Optional — increases GitHub API rate limits
```

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔧 Tools Reference (33 total)

### 🕐 Datetime & Utilities
| Tool                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `get_current_datetime` | Current date & time for any timezone             |
| `calculate`            | Safe math expression evaluator (AST-sandboxed)   |
| `convert_units`        | Length, weight, volume, temperature, speed, area |
| `analyze_text`         | Word/sentence/paragraph/readability statistics   |
| `generate_uuid`        | Generate unique UUIDs                            |
| `random_number`        | Random integers or floats within a range         |

### 🌦️ Weather & Location
| Tool              | Description                                  |
| ----------------- | -------------------------------------------- |
| `get_weather`     | Live weather for any city (Open-Meteo, free) |
| `geocode_address` | Address or place → lat/lon coordinates       |
| `get_timezone`    | Timezone and local time for any city         |
| `ip_geolocation`  | Geolocate any IP address (ip-api.com, free)  |

### 🔍 Search & Web
| Tool               | Description                               |
| ------------------ | ----------------------------------------- |
| `search_web`       | DuckDuckGo web search (no API key needed) |
| `fetch_webpage`    | Fetch and read any public URL             |
| `search_wikipedia` | Wikipedia summaries for any topic         |

### 📚 Knowledge & Research
| Tool                   | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `search_arxiv`         | Academic papers from arXiv (AI, CS, physics, math) |
| `search_pubmed`        | Medical/biomedical research via NCBI Entrez (free) |
| `stackoverflow_search` | Programming Q&A from Stack Overflow                |

### 💻 Code & Developer
| Tool            | Description                                            |
| --------------- | ------------------------------------------------------ |
| `run_python`    | Execute Python in a sandboxed subprocess (10s timeout) |
| `github_search` | Search GitHub repos, code, issues, users               |
| `lint_code`     | Syntax-check Python code                               |

### 💹 Finance & Crypto
| Tool                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `get_stock_price`      | Live stock price, P/E, market cap, 52-week range |
| `get_crypto_price`     | Crypto price, market cap, ATH (CoinGecko, free)  |
| `get_forex_rate`       | Live foreign exchange rates                      |
| `portfolio_calculator` | Multi-symbol portfolio total value               |

### 🗂️ File & Data
| Tool               | Description                              |
| ------------------ | ---------------------------------------- |
| `read_csv`         | Preview and analyze CSV files            |
| `extract_pdf_text` | Extract text from PDF files (pdfplumber) |
| `read_docx`        | Read Word (.docx) documents              |
| `query_sql`        | Run SELECT queries on SQLite databases   |

### 🖼️ Image
| Tool             | Description                                 |
| ---------------- | ------------------------------------------- |
| `resize_image`   | Resize any image file (Pillow)              |
| `get_image_info` | Get image dimensions, format, and file size |

### 🧠 AI Memory (persistent across sessions)
| Tool                        | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `remember_fact`             | Store a key/value fact in persistent memory         |
| `recall_memories`           | Retrieve all stored memories (with optional filter) |
| `forget_memory`             | Delete a specific memory by key                     |
| `search_past_conversations` | Full-text search on conversation titles             |

---

## 🧠 How Memory Works

Memories are stored in the `memories` table inside `chatbot.db` — the same SQLite file used for conversation checkpointing. This means memories persist across app restarts and conversations automatically.

**Example usage:**

> **You:** Remember that my name is Alex and I prefer metric units.

> **Bot:** ✅ Remembered: `user_name` → "Alex" and `unit_preference` → "metric"

> *(next session)*

> **You:** What unit system do I prefer?

> **Bot:** *(recalls memories)* You prefer metric units, Alex!

---

## 💬 Features

- **Multi-conversation sidebar** — create and switch between unlimited chat threads
- **Auto-titling** — conversation titles are generated from your first message
- **Live tool status** — see which tool is running in real time while the agent thinks
- **Collapsible tool results** — raw tool outputs shown in expandable panels
- **Streaming responses** — text streams token by token as it's generated
- **Persistent history** — conversations survive page reloads and app restarts
- **AI Memory** — the bot can remember facts you tell it across different sessions

---

## 🌍 API Keys & Costs

| Service                         | Key Required              | Cost                                |
| ------------------------------- | ------------------------- | ----------------------------------- |
| OpenAI (GPT-4o-mini)            | ✅ Yes — `OPENAI_API_KEY`  | ~$0.15 / 1M input tokens            |
| Open-Meteo (weather, geocoding) | ❌ No                      | Free                                |
| CoinGecko (crypto)              | ❌ No                      | Free                                |
| NCBI Entrez (PubMed)            | ❌ No                      | Free                                |
| Stack Exchange API              | ❌ No                      | Free                                |
| DuckDuckGo Search               | ❌ No                      | Free                                |
| ip-api.com                      | ❌ No                      | Free                                |
| yfinance (stocks, forex)        | ❌ No                      | Free                                |
| GitHub REST API                 | Optional — `GITHUB_TOKEN` | Free (higher rate limit with token) |

---

## 📌 Example Prompts

```
"What's the weather in Tokyo right now?"
"Search arXiv for recent papers on vision transformers"
"What's the Bitcoin price and how has it changed in 24h?"
"Run this Python code: print([x**2 for x in range(10)])"
"Read the file /data/sales.csv and summarize it"
"Remember that my preferred language is Python"
"What's 15% of 3,847 + sqrt(256)?"
"Search Stack Overflow for 'asyncio gather exception handling'"
"What's the current USD to NPR exchange rate?"
"What timezone is Kathmandu in and what time is it there?"
"Calculate my portfolio: AAPL:10, TSLA:5, GOOGL:2"
```

---

## 🔒 Security Notes

- `run_python` runs in an isolated subprocess with a 10-second hard timeout. Avoid exposing the app publicly without additional sandboxing.
- `query_sql` only allows `SELECT` statements — write operations are blocked.
- `calculate` uses AST parsing to block imports and attribute access — no `eval` on raw strings.

---

## 📄 License

MIT — free to use, modify, and distribute.