#!/usr/bin/env python3
import os
import sys
import json
import pickle
import shutil
from pathlib import Path
from typing import List

sys.path.append(os.getcwd())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.logger import logger

VECTOR_ROOT = Path(settings.VECTOR_DIR)
INDEX_DIR = VECTOR_ROOT / "faiss_index"
TEMP_INDEX_DIR = VECTOR_ROOT / "faiss_index_tmp"

SELECTED_JSON = Path("data/selected_pdfs.json")  # same as your script


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
    marker_count = sum(1 for m in TRANSCRIPT_MARKERS if m in t)
    if marker_count >= 1:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    colon_lines = sum(1 for ln in lines if ':' in ln and len(ln.split(':')[0]) <= 40)
    if colon_lines >= 3:
        return True
    if "question" in t and "answer" in t:
        return True
    return False


def load_selected_pdfs() -> List[dict]:
    if not SELECTED_JSON.exists():
        logger.error("selected_pdfs.json NOT FOUND at %s", SELECTED_JSON)
        return []
    data = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    pdfs = []
    for item in data:
        p = Path(item["path"])
        if p.exists() and p.is_file():
            item["path"] = str(p.resolve())
            pdfs.append(item)
        else:
            logger.warning("Missing PDF in selection list: %s", p)
    logger.info("Loaded %d selected PDFs", len(pdfs))
    return pdfs


def init_embedding_model():
    try:
        emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        logger.info("Using embedding model: %s", settings.EMBEDDING_MODEL_NAME)
        return emb
    except Exception as e:
        logger.exception("Failed to initialize HuggingFaceEmbeddings: %s", e)
        # Provide actionable hints
        msg = (
            "Embedding initialization failed. Common fixes:\n"
            "1) Ensure you installed sentence-transformers and compatible huggingface-hub.\n"
            "   Example (clean venv):\n"
            "     python -m venv .venv_rag && .\\.venv_rag\\Scripts\\Activate.ps1\n"
            "     pip install --upgrade pip\n"
            "     pip install sentence-transformers\n"
            "   OR pin hf hub (legacy): pip install 'huggingface-hub==0.25.2' 'sentence-transformers==2.2.2'\n"
            "2) If you use a shared env with gradio/diffusers/auto-gptq, create a dedicated venv for ingestion.\n"
        )
        logger.error(msg)
        raise RuntimeError("Embedding initialization failed; see logs for pip fixes.") from e


def main():
    selected = load_selected_pdfs()
    if not selected:
        logger.error("No PDFs to ingest.")
        return

    # Prepare splitter once
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Create temp index dir (atomic save later)
    TEMP_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        embed_model = init_embedding_model()
    except Exception:
        logger.error("Cannot continue without embeddings. Exiting.")
        return

    all_texts = []
    all_metas = []

    total_pages = 0
    processed_chunks = 0

    for item in selected:
        pdf_path = item["path"]
        pdf_name = item.get("name", Path(pdf_path).name)
        pdf_type = item.get("type", None)
        year, quarter = (None, None)
        qy = item.get("qy")
        if qy and isinstance(qy, (list, tuple)) and len(qy) >= 2:
            year, quarter = qy[0], qy[1]

        logger.info("Loading PDF: %s", pdf_path)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            total_pages += len(pages)

            # page-level classification and metadata
            for d in pages:
                d.metadata = d.metadata or {}
                page_text = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
                d.metadata["page_type"] = "transcript" if is_transcript_text(page_text) else None

            for d in pages:
                d.metadata = d.metadata or {}
                d.metadata["file_name"] = pdf_name
                d.metadata["source"] = pdf_path
                d.metadata["type"] = d.metadata.get("page_type") or pdf_type or "unknown"
                d.metadata["year"] = year
                d.metadata["quarter"] = quarter

            # chunk this file's pages immediately (memory-friendly)
            chunks = splitter.split_documents(pages)
            logger.info("File %s -> %d chunks", pdf_name, len(chunks))
            processed_chunks += len(chunks)

            # collect texts/meta for saving later
            all_texts.extend([c.page_content for c in chunks])
            all_metas.extend([c.metadata for c in chunks)

                              ]
        except Exception as e:
            logger.exception("Failed loading: %s : %s", pdf_path, e)
            continue

    if not all_texts:
        logger.error("No chunks created; exiting.")
        return

    logger.info("Total pages: %d; Total chunks: %d", total_pages, processed_chunks)

    # Build FAISS vectorstore (wrap to catch faiss / embedding errors)
    try:
        # create Documents-like objects for FAISS.from_documents — using same structure as LangChain expects
        # Here we reconstruct minimal document objects with .page_content and .metadata attributes
        class _D:
            def __init__(self, text, meta):
                self.page_content = text
                self.metadata = meta

        docs_for_faiss = [_D(t, m) for t, m in zip(all_texts, all_metas)]

        vectordb = FAISS.from_documents(docs_for_faiss, embed_model)
        # Save to temp dir
        vectordb.save_local(str(TEMP_INDEX_DIR))
        # pickle texts + meta
        with open(TEMP_INDEX_DIR / "texts.pkl", "wb") as f:
            pickle.dump(all_texts, f)
        with open(TEMP_INDEX_DIR / "meta.pkl", "wb") as f:
            pickle.dump(all_metas, f)

        # atomic replace
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        TEMP_INDEX_DIR.rename(INDEX_DIR)
        logger.info("Saved FAISS index + texts.pkl + meta.pkl to %s", INDEX_DIR)
        print("DONE: FAISS index created at:", INDEX_DIR)

    except Exception as e:
        logger.exception("Failed building/saving FAISS index: %s", e)
        logger.error(
            "Common causes: faiss library not installed, sentence-transformers/huggingface_hub mismatch, or insufficient RAM."
        )
        logger.error("If faiss is missing, try: pip install faiss-cpu")
        # cleanup temp dir if partial
        try:
            if TEMP_INDEX_DIR.exists():
                shutil.rmtree(TEMP_INDEX_DIR)
        except Exception:
            pass
        return


if __name__ == "__main__":
    main()
