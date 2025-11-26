# scripts/step2_ingest_rag_fixed.py
import os
import sys
import json
import pickle
import logging
from pathlib import Path

# keep repo root on sys.path if script is run directly (safe)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# your app settings + logger
from app.config import settings
from app.logger import logger

log = logger  # keep naming consistent below

# -----------------------------------------------------------
# Vector / index locations (use settings.VECTOR_DIR as source of truth)
# -----------------------------------------------------------
VECTOR_ROOT = Path(settings.VECTOR_DIR)
INDEX_DIR = VECTOR_ROOT / "faiss_index"

# -----------------------------------------------------------
# SELECTED PDF list: prefer an explicit setting, fallback to common locations
# -----------------------------------------------------------
def resolve_selected_json():
    # 1) explicit in settings (recommended)
    cand = None
    if hasattr(settings, "SELECTED_PDFS") and settings.SELECTED_PDFS:
        cand = Path(settings.SELECTED_PDFS)
    # 2) env override
    if not cand:
        env_val = os.environ.get("SELECTED_PDFS")
        if env_val:
            cand = Path(env_val)
    # 3) common repo-local location (project-root relative). Try to find project root.
    if not cand:
        # try to guess project root from installed app package (fall back to cwd)
        try:
            # If this file is in scripts/, parents[1] is repo root
            repo_root = Path(__file__).resolve().parents[1]
        except Exception:
            repo_root = Path.cwd()
        cand = repo_root / "data" / "selected_pdfs.json"

    # resolve and return
    return cand.resolve()

SELECTED_FILE = resolve_selected_json()

# -----------------------------------------------------------
# Load selected_pdfs.json (only paths that exist)
# -----------------------------------------------------------
def load_selected_pdfs():
    if not SELECTED_FILE.exists():
        log.error("selected_pdfs.json NOT FOUND at %s", SELECTED_FILE)
        return []

    try:
        data = json.loads(SELECTED_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.exception("Failed to read/parse selected_pdfs.json: %s", e)
        return []

    pdfs = []
    for item in data:
        p = Path(item.get("path", "") or "")
        # If relative path provided, try resolving relative to SELECTED_FILE parent
        if p and not p.is_absolute():
            p = (SELECTED_FILE.parent / p).resolve()

        if p.exists() and p.is_file():
            item["path"] = str(p.resolve())
            pdfs.append(item)
        else:
            log.warning("Missing PDF in selection list: %s", p)

    log.info("Loaded %d selected PDFs from %s", len(pdfs), SELECTED_FILE)
    return pdfs


# -----------------------------------------------------------
# Simple page-level transcript detector (as you had)
# -----------------------------------------------------------
TRANSCRIPT_MARKERS = [
    "operator", "question and answer", "q&a", "earnings conference call",
    "earnings call", "moderator:", "thank you", "open for questions",
    "tata consultancy services earnings conference call", "investor call",
    "investor conference", "participants", "question from", "panelist"
]


def is_transcript_text(txt: str) -> bool:
    if not txt:
        return False
    t = txt.lower()

    # marker occurrence count
    marker_count = sum(1 for m in TRANSCRIPT_MARKERS if m in t)
    if marker_count >= 1:
        return True

    # many lines with colon (Name: speech) usually indicates transcript
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    colon_lines = sum(1 for ln in lines if ':' in ln and len(ln.split(':')[0]) <= 40)
    if colon_lines >= 3:
        return True

    # presence of "question" and "answer" together strongly indicates a transcript
    if "question" in t and "answer" in t:
        return True

    return False


# -----------------------------------------------------------
# MAIN INGEST PROCESS
# -----------------------------------------------------------
def main():
    selected = load_selected_pdfs()
    if not selected:
        log.error("No PDFs to ingest.")
        return

    docs = []

    for item in selected:
        pdf_path = item["path"]
        pdf_name = item.get("name", Path(pdf_path).name)
        pdf_type = item.get("type", None)  # factsheet / transcript / other
        year, quarter = (None, None)
        try:
            qy = item.get("qy")
            if qy and isinstance(qy, (list, tuple)) and len(qy) >= 2:
                year, quarter = qy[0], qy[1]
        except Exception:
            pass

        log.info("Loading PDF: %s", pdf_path)

        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()

            # classify each page as transcript/factsheet/other using page text
            for d in pages:
                if d.metadata is None:
                    d.metadata = {}

                page_text = d.page_content if hasattr(d, "page_content") else (d.content if hasattr(d, "content") else "")
                d.metadata["page_type"] = "transcript" if is_transcript_text(page_text) else None

            # Add metadata to each document and prefer page_type over file-level type
            for d in pages:
                d.metadata = d.metadata or {}
                d.metadata["file_name"] = pdf_name
                d.metadata["source"] = pdf_path
                d.metadata["type"] = d.metadata.get("page_type") or pdf_type or "unknown"
                d.metadata["year"] = year
                d.metadata["quarter"] = quarter

            docs.extend(pages)

        except Exception as e:
            log.exception("Failed loading: %s : %s", pdf_path, e)

    if not docs:
        log.error("No documents extracted!")
        return

    # ---------------------------------------------
    # Chunking
    # ---------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    log.info("Split into %d chunks", len(chunks))

    # ---------------------------------------------
    # Embed + FAISS
    # ---------------------------------------------
    try:
        embed_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    except Exception as e:
        log.exception("Failed to initialize embedding model '%s': %s", getattr(settings, "EMBEDDING_MODEL_NAME", None), e)
        return

    # ensure index dir
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        vectordb = FAISS.from_documents(chunks, embed_model)
        vectordb.save_local(str(INDEX_DIR))
    except Exception as e:
        log.exception("Failed to build/save FAISS index: %s", e)
        return

    # ---------------------------------------------
    # Save aligned texts + metadata
    # ---------------------------------------------
    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]

    try:
        with open(INDEX_DIR / "texts.pkl", "wb") as f:
            pickle.dump(texts, f)
        with open(INDEX_DIR / "meta.pkl", "wb") as f:
            pickle.dump(metas, f)
    except Exception as e:
        log.exception("Failed to write texts/meta pkl files: %s", e)
        return

    log.info("Saved FAISS index + texts.pkl + meta.pkl to %s", INDEX_DIR)
    print("DONE: FAISS index created at:", INDEX_DIR)


if __name__ == "__main__":
    main()
