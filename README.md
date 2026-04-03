# AI Leadership Insight Agent

A production-ready RAG (Retrieval-Augmented Generation) agent that answers
executive questions grounded in your organisation's internal documents.

---

## Architecture

```
documents/           ← Your PDF, DOCX, TXT files
    │
    ▼
[DocumentIngester]   ingestion.py
    │  • Loads PDF / DOCX / TXT
    │  • Cleans & normalises text
    │  • Splits into overlapping chunks (default 800 words, 100 overlap)
    ▼
[VectorStore]        vector_store.py
    │  • Embeds chunks with sentence-transformers (all-MiniLM-L6-v2, ~23 MB)
    │  • Indexes with FAISS IndexFlatIP (cosine similarity on L2-normalised vecs)
    │  • Persists to disk; loads on subsequent runs (no re-embedding needed)
    ▼
[LeadershipInsightAgent]  agent.py
    │  • Receives a natural-language question
    │  • Retrieves top-k relevant chunks from FAISS
    │  • Calls Claude claude-sonnet-4-20250514 with document context + question
    │  • Returns a cited, factual answer
    ▼
main.py              CLI entry-point (interactive or single-question mode)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Add your company documents

Drop PDF, DOCX, or TXT files into the `documents/` folder.
Two sample documents are included for immediate testing.

### 4. Run

**Interactive mode** (recommended for exploration):
```bash
python main.py
```

**Single question** (great for scripting / CI):
```bash
python main.py --question "What is our current revenue trend?"
```

**Force re-index after adding new documents:**
```bash
python main.py --reload
```

**JSON output** (for downstream processing):
```bash
python main.py --question "Which departments are underperforming?" --json
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --docs PATH       Documents directory (default: ./documents)
  --index PATH      FAISS index directory (default: ./vector_index)
  --question/-q     Single question to answer
  --reload          Force re-ingestion of all documents
  --top-k INT       Chunks retrieved per query (default: 6)
  --json            Output result as JSON
```

---

## Supported Document Types

| Format | Library       |
|--------|---------------|
| `.txt` | built-in      |
| `.md`  | built-in      |
| `.pdf` | pypdf         |
| `.docx`| python-docx   |

---

## Example Questions

```
"What is our current revenue trend?"
"Which departments are underperforming?"
"What were the key risks highlighted in the last quarter?"
"What is our full-year revenue guidance?"
"What is our current employee attrition rate and what are the root causes?"
"What acquisition targets are we pursuing?"
"How are EMEA and APAC regions performing?"
```

---

## Programmatic API

```python
from agent import LeadershipInsightAgent

agent = LeadershipInsightAgent(
    docs_dir="./documents",
    index_path="./vector_index",
    top_k=6,
)
agent.ingest_documents()

result = agent.ask("What are the key risks in Q3 2024?")
print(result["answer"])
print("Sources:", result["sources"])
```

`result` schema:
```json
{
  "question": "...",
  "answer":   "...",
  "sources":  ["q3_2024_quarterly_report.txt", "..."],
  "context_chunks": 6
}
```

---

## Configuration Tuning

| Parameter      | Default | Notes |
|----------------|---------|-------|
| `chunk_size`   | 800     | Words per chunk. Larger = more context per chunk; smaller = more precise retrieval |
| `chunk_overlap`| 100     | Words shared between consecutive chunks. Prevents boundary information loss |
| `top_k`        | 6       | Chunks retrieved per query. Increase for broad questions; decrease for speed |

---

## Streamlit UI

`LeadershipInsightAgent` now accepts optional **`llm_provider`** and per-provider keys (`anthropic_api_key`, `openai_api_key`, `groq_api_key`, `groq_base_url`, `ollama_base_url`, `ollama_api_key`).  
If **`llm_provider` is `None`**, behavior matches before (env + auto-priority).  
If **`llm_provider` is set** (e.g. from the UI), the right key is required; whitespace is trimmed.

### **`streamlit_app.py`** (new)

- **Wide layout**, dark gradient background, **Plus Jakarta Sans** + **Fraunces** for the title  
- **Sidebar:** provider (Groq, Ollama, Anthropic, OpenAI), **password API key**, **model** dropdown + **“Custom model…”**, **top‑k** slider  
- **Documents:** bundled examples, **project folder**, or **multi-file upload** (PDF/DOCX/TXT)  
- **Advanced:** index path, **rebuild index**, optional Groq base URL  
- **Apply & load documents** builds the agent, runs ingestion, then enables chat  
- **Chat UI** with `st.chat_input`, source expanders, errors surfaced without crashing the app  
- **`st.secrets`** support for default keys (`anthropic_api_key`, `openai_api_key`, `groq_api_key`) on Streamlit Cloud  

### **`requirements.txt`**

- Added `streamlit>=1.40.0`

### Run

```powershell
cd C:\Users\Manikanta\Downloads\leadership_agent
pip install -r requirements.txt
streamlit run streamlit_app.py

## Task 2 Extension Points

To evolve this into an **autonomous decision-making agent** (Task 2):

1. **Web search tool** — add Anthropic's tool-use API to let the agent fetch live
   market/competitor data from the web.
2. **Multi-step reasoning** — implement a ReAct loop: the agent reasons, selects a
   tool, observes the result, then reasons again until it has a final answer.
3. **Action layer** — connect to Slack / email / Jira to let the agent draft
   recommendations, create action items, or escalate risks automatically.
4. **Conversation memory** — persist the Q&A history and allow follow-up questions
   that reference earlier answers.
5. **Scheduled ingestion** — add a cron job or file-watcher to automatically
   re-index whenever new documents land in the folder.
