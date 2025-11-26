📘 Financial Forecasting Agent for TCS

An end-to-end AI-driven financial forecasting agent built using FastAPI, LangChain, FAISS RAG, and OpenRouter LLMs.
The system automatically retrieves quarterly financial documents for Tata Consultancy Services (TCS), extracts structured financial metrics, performs qualitative transcript analysis, and generates a machine-readable business outlook forecast.

This README explains the architecture, tools, prompt design, setup steps, and run instructions — exactly as required in the Elevation AI Assessment.

🏗️ 1. Project Overview
🎯 Goal

Build an AI agent that can:

Automatically fetch all financial documents (fact sheets + transcripts) from Screener.in

Download + fix PDFs/ASPX

Select the latest 1–3 quarters

Build a RAG vector database for semantic transcript search

Extract quantitative financial metrics

Perform qualitative analysis (themes, sentiment, outlook)

Generate a structured JSON forecast

Serve everything through FastAPI

Log requests + responses to MySQL

Everything is automated — no manual file downloading required.

🧠 2. Architectural Approach

The entire system follows a Tool-Driven Agent Architecture:

   [ Screener Pipeline ]
           ↓
[ Financial Extractor ]    [ Qualitative RAG ]
           ↓                       ↓
   [ Agent Orchestrator + LangChain ]
                       ↓
               [ LLM Forecast JSON ]
                       ↓
           [ FastAPI Endpoint + MySQL Logging ]

2.1 Screener Pipeline (Automated Document Retrieval)

File: app/tools/screener_pipeline.py

Runs 5 scripts in order:

scrape_screener_pdfs.py – Find all document URLs

download_all_pdfs.py – Download every PDF/ASPX

fix_aspx_files.py – Convert/repair ASPX files

last_3_quarters.py – Detect latest 1–3 quarters

rag_vectorDB.py – Build FAISS vector DB

Outputs:

Cleaned PDFs → data/docs/screener_pdfs/

Quarter selection → data/selected_pdfs.json

FAISS store → faiss_index, meta.pkl, texts.pkl

2.2 FinancialDataExtractorTool

File: app/tools/financial_extractor.py

Extracts:

Revenue

Net Profit

Operating Margin

YoY %

Evidence snippets

Patterns + raw matches

Uses:

pdfplumber parsing

Regex-based metric identification

Error-safe extraction

Returns structured dict for LLM

2.3 QualitativeAnalysisTool (RAG)

File: app/tools/qualitative_tool.py

Performs:

SentenceTransformer embeddings (MiniLM-L6-v2)

Top-K semantic retrieval from transcripts

Sentiment analysis (DistilBERT SST-2)

Forward-looking statement extraction

Recurring theme detection

Topic-based highlighting (demand, attrition, guidance, etc.)

Produces:

Themes

Sentiment per document

Important snippets

Forward-looking statements

Topic-based hits

2.4 Agent Orchestrator

File: app/agent_orchestrator.py

Responsibilities:

Combine:

financial extractor output

qualitative RAG output

Auto-trim large evidence to avoid token overflow

Build structured LangChain prompt

Send to OpenRouter LLM

Enforce strict JSON output

Parse & validate JSON

Return final response

2.5 FastAPI Service

File: app/api_fastapi.py

Endpoints:

GET /           → Health check
GET /forecast   → Generates forecast JSON
POST /forecast  → Future extension for uploads


Includes:

Background MySQL logging

Automatic cache warmup

Automatic Screener pipeline (optional)

2.6 MySQL Logging

File: app/db/database.py, app/db/models.py

Every request/response pair is logged:

request_payload

response_payload

forecast status

source file reference

timestamp

Stored in:

forecast_logs

🎛️ 3. Agent & Tool Design
3.1 Tool Usage in the LLM Chain

The LLM receives a merged payload:

{
  "financial": {...},
  "qualitative": {...},
  "generated_at": timestamp
}


It then synthesizes:

financial trends

management outlook

risks & opportunities

next-quarter revenue forecast

confidence level

evidence list

3.2 Master Prompt

The system prompt (strict JSON enforcement):

You are a financial forecasting agent. Return ONLY valid JSON in this exact format:

{
  "summary": "",
  "financial_trends": {},
  "management_outlook": "",
  "risks": [],
  "opportunities": [],
  "forecast": {
    "revenue_growth_next_quarter_pct": 0.0,
    "margin_trend": "",
    "confidence": ""
  },
  "evidence": [],
  "generated_at": "ISO timestamp"
}

Use extracted financial metrics + qualitative transcript findings.
If a value is missing, set null. No explanations. JSON only.


This ensures the LLM:

never returns plain text

stays deterministic

outputs predictable structured data

⚙️ 4. Setup Instructions (Must Follow Exactly)
⭐ Step 1 — Clone the Repository
git clone <your-github-url>
cd Forecast_agent_TCS

⭐ Step 2 — Create Virtual Environment
Windows:
python -m venv venv
.\venv\Scripts\Activate.ps1

macOS/Linux:
python3 -m venv venv
source venv/bin/activate

⭐ Step 3 — Install Dependencies
pip install -r requirements.txt

⭐ Step 4 — Configure MySQL 8.0

Run:

& "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p


Inside MySQL:

CREATE DATABASE forecast_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'forecast'@'localhost' IDENTIFIED BY 'your_password';

GRANT ALL ON forecast_agent.* TO 'forecast'@'localhost';
FLUSH PRIVILEGES;

⭐ Step 5 — Set Environment Variables
$env:FORECAST_DB_URL="mysql+pymysql://forecast:your_password@127.0.0.1:3306/forecast_agent"
$env:OPENROUTER_API_KEY="sk-your-key"
$env:OPENROUTER_BASE="https://openrouter.ai/api/v1"

⭐ Step 6 — Initialize DB Tables
python -c "from app.db.database import init_db; init_db(); print('DB initialized')"

⭐ Step 7 — Start FastAPI Server
uvicorn app.api_fastapi:app --reload --port 8000


Startup will show:

Running Screener ingestion pipeline...
scrape_ok
download_ok
fix_aspx_ok
select_ok
ingest_ok
Cache warmup complete.

🧪 5. How to Run the Agent
Generate a forecast:
Invoke-RestMethod http://localhost:8000/forecast


or:

curl http://localhost:8000/forecast

MySQL logs:
USE forecast_agent;
SELECT * FROM forecast_logs ORDER BY id DESC LIMIT 5;

Run Screener Pipeline Manually:
python -c "from app.tools.screener_pipeline import ScreenerPipeline; print(ScreenerPipeline.run())"

📂 6. Repository Structure
app/
  api_fastapi.py
  agent_orchestrator.py
  cache_manager.py
  tools/
    financial_extractor.py
    qualitative_tool.py
    screener_pipeline.py
  db/
    database.py
    models.py
scripts/
  scrape_screener_pdfs.py
  download_all_pdfs.py
  fix_aspx_files.py
  last_3_quarters.py
  rag_vectorDB.py
data/
  docs/
  selected_pdfs.json
requirements.txt
README.md
LICENSE
.gitignore

