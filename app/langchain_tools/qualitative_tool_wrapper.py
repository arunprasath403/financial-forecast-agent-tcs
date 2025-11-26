# app/langchain_tools/qualitative_tool_wrapper.py

from langchain.tools import tool
from typing import Optional, Dict, Any

# Import your actual qualitative analyzer
from app.tools.qualitative_tool import QualitativeAnalysisTool

@tool("qualitative_analyzer", return_direct=True)
def qualitative_analyzer_tool(topics: Optional[str] = None) -> Dict[str, Any]:
    """
    LangChain wrapper around QualitativeAnalysisTool.
    """
    tool = QualitativeAnalysisTool()
    
    if topics:
        topics_list = [t.strip() for t in topics.split(",")]
        return tool.analyze(topics=topics_list)
    
    return tool.analyze()
