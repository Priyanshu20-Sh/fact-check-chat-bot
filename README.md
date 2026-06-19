# Fact-Checking Web App ("Truth Layer")
# WARNING - Ensure your environment has internet access for Tavily Search and LLM endpoints.

A Fact-Checking assistant built with Streamlit and LangGraph. It acts as a "Truth Layer," extracting claims from a provided PDF document, cross-referencing them against live web data, and flagging inaccuracies.

Instead of just answering questions based on the document, the agent verifies the document's statistics and facts against the real world.

---

# What's in here?

1 - Smart Routing - Evaluates questions to decide whether to query local docs for fact-checking, search the web directly, or use general LLM knowledge.

2 - Hybrid Retrieval - Combines FAISS (dense semantic vectors) and BM25 (sparse keyword matches) to retrieve claims from your uploaded document.

3 - Verification Agent - Takes the claims extracted from your document and automatically cross-references them against live web search using Tavily.

4 - Fact-Check Reporting - Flags the document's claims as Verified, Inaccurate, or False, and provides the real facts found on the web.

## Quick Setup (using uv)

### 1. Extract & Navigate
```bash
cd Autonomus_Rag
```

### 2. Install Dependencies
Since this project is managed with `uv`, you can spin up the environment and install all dependencies with a single command:
```bash
uv sync
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY="your-groq-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

---

## How to Run It

You can run the app directly using `uv run`:
```bash
uv run streamlit run app.py
```

Or activate the environment first:
```bash
# On Windows (PowerShell)
.venv\Scripts\activate
python -m streamlit run app.py
```
