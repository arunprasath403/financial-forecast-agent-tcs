# app/api_fastapi.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from fastapi import BackgroundTasks
from app.db.logger import log_request_and_response
from app.agent_orchestrator import run_langchain_agent
from app.cache_manager import CacheManager


# ----------------------------
# CREATE APP FIRST (IMPORTANT)
# ----------------------------
app = FastAPI(
    title="TCS Financial Forecast Agent API",
    description="AI-powered forecast agent using OpenRouter + LangChain tools",
    version="1.0.0"
)


# ----------------------------
# STARTUP EVENT (CACHE)
# ----------------------------
@app.on_event("startup")
def load_cache():
    CacheManager.warm_up()


# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# ROOT
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "TCS Forecast Agent API is running.",
        "timestamp": datetime.utcnow().isoformat()
    }


# ----------------------------
# FORECAST (GET)
# ----------------------------



@app.get("/forecast")
def forecast_get(background_tasks: BackgroundTasks):
    """
    GET-based forecast generation using cached data + background DB logging
    """
    try:
        result = run_langchain_agent()

        # timestamp (if not already added)
        if "generated_at" not in result:
            result["generated_at"] = datetime.utcnow().isoformat()

        # Non-blocking DB logging (fast)
        background_tasks.add_task(
            log_request_and_response,
            request_payload={},                           # you can store more metadata later
            response_payload=result,
            source_file_path="D:\\Forecast_agent_TCS\\app\\tools\\financial_extractor.py"
        )

        return result

    except Exception as e:

        # if error, log error row too (non-blocking)
        background_tasks.add_task(
            log_request_and_response,
            request_payload={},
            response_payload=None,
            source_file_path="D:\\Forecast_agent_TCS\\app\\tools\\financial_extractor.py",
            status="error",
            error_text=str(e)
        )

        raise HTTPException(status_code=500, detail=str(e))



# ----------------------------
# FORECAST (POST)
# ----------------------------
@app.post("/forecast")
def forecast_post(payload: dict = {}):
    try:
        result = run_langchain_agent()

        if "generated_at" not in result:
            result["generated_at"] = datetime.utcnow().isoformat()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
