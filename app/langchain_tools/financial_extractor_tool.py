# app/langchain_tools/financial_extractor_tool.py

from langchain.tools import tool
from typing import Optional, Dict, Any

# Import your actual tool file
from app.tools.financial_extractor import extract_financial_metrics

@tool("financial_extractor", return_direct=True)
def financial_extractor_tool(pdf_path: Optional[str] = None) -> Dict[str, Any]:
    """
    LangChain wrapper around your financial extractor.
    If pdf_path is None, it will use FAISS/texts fallback.
    """
    return extract_financial_metrics(pdf_path)
