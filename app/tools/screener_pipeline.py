# app/tools/screener_pipeline.py

import subprocess
import sys
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger("tcs_agent.screener_pipeline")

# --- Adjust script paths based on your project structure ---
ROOT = Path(__file__).resolve().parents[2]  # D:\Forecast_agent_TCS
SCRIPTS_DIR = ROOT / "scripts"

SCRAPE = SCRIPTS_DIR / "scrape_screener_pdfs.py"      # Step 1
DOWNLOAD = SCRIPTS_DIR / "download_pdfs.py"            # Step 2
FIX_ASPX = SCRIPTS_DIR / "fix_aspx_files.py"           # Step 3
SELECT = SCRIPTS_DIR / "last_3_quarters.py"            # Step 4
INGEST = SCRIPTS_DIR / "rag_vectorDB.py"               # Step 5

# Output paths
OUT_SELECT_JSON = ROOT / "data" / "selected_pdfs.json"
OUT_PDF_DIR = ROOT / "data" / "docs" / "screener_pdfs"
OUT_LINKS = ROOT / "data" / "docs" / "screener_links.txt"

PY = sys.executable


class ScreenerPipeline:
    """Runs all 5 Screener ingestion scripts in correct order."""

    @staticmethod
    def _run_script(path: Path, timeout=300):
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")

        cmd = [PY, str(path)]
        logger.info("Running: %s", cmd)

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )

        logger.info("stdout:\n%s", process.stdout[:2000])
        if process.returncode != 0:
            logger.error("stderr:\n%s", process.stderr[:2000])
            raise RuntimeError(f"Script failed: {path.name}")

        return process.stdout

    @classmethod
    def run(cls) -> Dict[str, Any]:
        """
        Runs the full Screener pipeline:
            1. Scrape links
            2. Download PDFs
            3. Fix ASPX files
            4. Select last 3 quarters (write selected_pdfs.json)
            5. Build FAISS RAG Vector DB

        Returns metadata summary.
        """

        summary = {"ok": True, "steps": []}

        try:
            # STEP 1: Scrape links
            cls._run_script(SCRAPE)
            summary["steps"].append("scrape_ok")

            # STEP 2: Download PDFs
            cls._run_script(DOWNLOAD)
            summary["steps"].append("download_ok")

            # STEP 3: Fix ASPX
            cls._run_script(FIX_ASPX)
            summary["steps"].append("fix_aspx_ok")

            # STEP 4: Select top 3 quarters
            cls._run_script(SELECT)
            summary["steps"].append("select_ok")

            # STEP 5: Build Vector DB
            cls._run_script(INGEST)
            summary["steps"].append("ingest_ok")

            # Validate outputs
            summary["selected_json"] = str(OUT_SELECT_JSON) if OUT_SELECT_JSON.exists() else None
            summary["pdf_count"] = len(list(OUT_PDF_DIR.glob("*.pdf"))) if OUT_PDF_DIR.exists() else 0

            return summary

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            return {"ok": False, "error": str(e), "steps": summary.get("steps")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ScreenerPipeline.run()
    print(result)
