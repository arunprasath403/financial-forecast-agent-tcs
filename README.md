# 📘 Financial Forecasting Agent for TCS

An **end-to-end AI-driven financial forecasting agent** built using **FastAPI, LangChain, FAISS RAG, and OpenRouter LLMs**.

The system automatically retrieves quarterly TCS financial documents, extracts quantitative + qualitative insights, and generates a strict **machine-readable JSON forecast**.

---

# 🧠 System Architecture

Below is the full **architecture diagram** (Mermaid):

```mermaid
flowchart TD

A[Screener.in<br/>Document Fetcher] --> B[PDF/ASPX Downloader]
B --> C[ASPX Fixer<br/>PDF Repair]
C --> D[Quarter Selector<br/>Latest 1–3 Qs]
D --> E[FAISS RAG Builder]

E --> G[Qualitative Analysis Tool<br/>Embeddings + Sentiment]
F[Financial Extractor<br/>PDF Metric Parser] --> H

G --> H[Agent Orchestrator<br/>LangChain + Prompting]
H --> I[OpenRouter LLM<br/>Strict JSON Output]
I --> J[FastAPI Endpoint /forecast]
J --> K[MySQL Logger<br/>forecast_logs]
```

---

# 🏗️ 1. Project Overview

### 🎯 Goal  
A fully automated pipeline that:

- Scrapes **TCS fact sheets + transcripts**
- Downloads + repairs **PDF/ASPX**
- Selects **latest financial quarters**
- Builds **FAISS RAG DB**
- Extracts **financial KPIs**
- Performs **qualitative outlook analysis**
- Generates **LLM-based forecast JSON**
- Logs everything in **MySQL**

No manual downloading.

---

# 🧩 2. Components

## 2.1 Screener Pipeline  
Location: `app/tools/screener_pipeline.py`

- scrape_screener_pdfs.py  
- download_all_pdfs.py  
- fix_aspx_files.py  
- last_3_quarters.py  
- rag_vectorDB.py  

Outputs:

- Clean PDFs → `data/docs/screener_pdfs/`
- selected_pdfs.json  
- FAISS DB → `faiss_index/`

---

## 2.2 Financial Extractor  
Location: `app/tools/financial_extractor.py`

Extracts:

- Revenue  
- Net Profit  
- Margins  
- YoY %  
- Evidence snippets  

Uses pdfplumber + regex.

---

## 2.3 Qualitative RAG  
Location: `app/tools/qualitative_tool.py`

Provides:

- Semantic retrieval
- Themes
- Sentiment score
- Forward-looking statements
- Topic highlights

---

## 2.4 Agent Orchestrator  
Location: `app/agent_orchestrator.py`

- Merges financial + qualitative signals  
- Builds LangChain prompt  
- Calls OpenRouter LLM  
- Enforces strict JSON  
- Returns final forecast  

---

## 2.5 FastAPI Service  
Location: `app/api_fastapi.py`

Endpoints:

- `/`  
- `/forecast`  

Handles:

- Startup cache  
- Optional Screener pipeline  
- MySQL logging  

---

# ⚙️ 3. Setup Instructions

## Step 1 — Clone (Use PowerShell)
```bash
git clone https://github.com/arunprasath403/financial-forecast-agent-tcs.git
cd financial-forecast-agent-tcs
```

## Step 2 — Venv
Windows:
```bash
python -m venv venv
.env\Scriptsctivate
```

Linux/macOS:
```bash
python3 -m venv venv
venv\Scripts\Activate
```

## Step 3 — Install
```bash
pip install -r requirements.txt
```

## Step 4 — MySQL
```sql
CREATE DATABASE forecast_agent;
CREATE USER 'forecast'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL ON forecast_agent.* TO 'forecast'@'localhost';
```

## Step 5 — Env Variables
```bash
FORECAST_DB_URL=mysql+pymysql://forecast:your_password@127.0.0.1:3306/forecast_agent
OPENROUTER_API_KEY=sk-your-key
OPENROUTER_BASE=https://openrouter.ai/api/v1
```

## Step 6 — Init DB
```bash
python -c "from app.db.database import init_db; init_db()"
```

## Step 7 — Run Server
```bash
uvicorn app.api_fastapi:app --reload --port 8000
```

---

# 🧪 4. Run the Agent

```bash
curl http://localhost:8000/forecast
```

---

# 📂 5. Repo Structure

```
app/
  api_fastapi.py
  agent_orchestrator.py
  cache_manager.py
  tools/
  db/
scripts/
data/
README.md
LICENSE
.gitignore
```

---
