from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
load_dotenv()  # loads .env from project root

class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MYSQL_URI: str = os.getenv("MYSQL_URI", "mysql+pymysql://root:password@127.0.0.1:3306/tcs_agent")
    VECTOR_DIR: str = os.getenv("VECTOR_DIR", "./data/vector_store")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data/docs")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    PDF_UPLOAD_PATH: str = os.getenv("PDF_UPLOAD_PATH", "D:/Forecast_agent_TCS/data/docs/screener_pdfs")

settings = Settings()
