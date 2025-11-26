#!/usr/bin/env python3
"""
scripts/rag_vectorDB.py  -- LangChain removed version

- Reads data/selected_pdfs.json
- Extracts text per PDF page using pypdf
- Heuristically detects transcript pages
- Splits text into overlapping character chunks
- Embeds chunks with SentenceTransformer
- Builds FAISS index (ID map) and writes index + texts.pkl + meta.pkl

Usage:
  Ensure venv has: pypdf, sentence-transformers, faiss-cpu, numpy
"""
import os
import sys
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# keep project root on path
sys.path.append(os.getcwd())

# third-party imports
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# local imports - keep your settings/logger
from app.config import settings
from app.logger import logger

# paths
VECTOR_ROOT = Path(settings.VECTOR_DIR)
INDEX_DIR = VECTOR_ROOT / "faiss_index"
TEMP_INDEX_DIR = VECTOR_ROOT / "faiss_index_tmp"
SELECTED_JSON = Path("data/selected_pdfs.json")

# transcript detection heuristics (same as before)
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
    if any(m in t for m in TRANSCRIPT_MARKERS):
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    colon_lines = sum(1 for ln in lines if ':' in ln and len(ln.split(':')[0]) <= 40)
    if colon_lines >= 3:
        return True
    if "question" in t and "answer" in t:
        return True
    return False


def load_selected_pdfs() -> List[Dict[str, Any]]:
    if not SELECTED_JSON.exists():
        logger.error("selected_pdfs.json NOT FOUND at %s", SELECTED_JSON)
        return []
    try:
        data = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("Failed reading selected_pdfs.json: %s", e)
        return []
    pdfs = []
    for item in data:
        p = Path(item.get("path", ""))
        if p.exists() and p.is_file():
            item["path"] = str(p.resolve())
            pdfs.append(item)
        else:
            logger.warning("Missing PDF in selection list: %s", p)
    logger.info("Loaded %d selected PDFs", len(pdfs))
    return pdfs


def extract_pages_text(pdf_path: str) -> List[str]:
    """
    Return list of page texts for the given PDF path using pypdf.
    If page extraction fails for a page, returns empty string for that page (so indexing remains aligned).
    """
    texts: List[str] = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            texts.append(txt)
    except Exception as e:
        logger.exception("Failed reading PDF %s: %s", pdf_path, e)
    return texts


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Character-based chunker with overlap. Returns list of chunk strings.
    """
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    start = 0
    chunks = []
    length = len(text)
    if length <= chunk_size:
        return [text]
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def main():
    selected = load_selected_pdfs()
    if not selected:
        logger.error("No PDFs to ingest.")
        return

    chunk_size = 1000
    chunk_overlap = 200

    # prepare temp dir
    if TEMP_INDEX_DIR.exists():
        try:
            shutil.rmtree(TEMP_INDEX_DIR)
        except Exception:
            pass
    TEMP_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # init embed model
    try:
        embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Using embedding model: %s", settings.EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.exception("Failed to initialize SentenceTransformer: %s", e)
        logger.error("Install sentence-transformers and check EMBEDDING_MODEL_NAME.")
        return

    all_texts: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    total_pages = 0
    for item in selected:
        pdf_path = item["path"]
        pdf_name = item.get("name", Path(pdf_path).name)
        pdf_type = item.get("type", None)
        year, quarter = (None, None)
        qy = item.get("qy")
        if qy and isinstance(qy, (list, tuple)) and len(qy) >= 2:
            year, quarter = qy[0], qy[1]

        logger.info("Loading PDF: %s", pdf_path)
        pages = extract_pages_text(pdf_path)
        total_pages += len(pages)

        # per-page classify & chunk
        for page_idx, page_text in enumerate(pages):
            page_meta = {
                "file_name": pdf_name,
                "source": pdf_path,
                "page_number": page_idx + 1,
                "type": None,
                "year": year,
                "quarter": quarter,
            }
            # detect transcript at page level
            page_meta["type"] = "transcript" if is_transcript_text(page_text) else (pdf_type or "unknown")

            # chunk page_text
            chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                # still add an empty chunk so indexes align? skip empty pages to save space
                continue

            for ch in chunks:
                all_texts.append(ch)
                all_metas.append(page_meta.copy())

    if not all_texts:
        logger.error("No chunks created; exiting.")
        return

    logger.info("Total pages: %d; Total chunks: %d", total_pages, len(all_texts))

    # create embeddings (batch)
    try:
        # encode returns float32 numpy array
        embeddings = embed_model.encode(all_texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
        # ensure 2D ndarray
        embeddings = np.asarray(embeddings, dtype=np.float32)
    except Exception as e:
        logger.exception("Failed computing embeddings: %s", e)
        return

    # build FAISS index (cosine similarity via inner product with normalized vectors)
    try:
        dim = embeddings.shape[1]
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index_flat = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
        index = faiss.IndexIDMap(index_flat)  # allow explicit ids
        ids = np.arange(0, embeddings.shape[0]).astype(np.int64)
        index.add_with_ids(embeddings, ids)
    except Exception as e:
        logger.exception("Failed to build FAISS index: %s", e)
        return

    # save index + texts + meta
    try:
        # write faiss index
        faiss_index_file = TEMP_INDEX_DIR / "index.faiss"
        faiss.write_index(index, str(faiss_index_file))

        # write texts/meta
        with open(TEMP_INDEX_DIR / "texts.pkl", "wb") as f:
            pickle.dump(all_texts, f)
        with open(TEMP_INDEX_DIR / "meta.pkl", "wb") as f:
            pickle.dump(all_metas, f)

        # atomic move
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        TEMP_INDEX_DIR.rename(INDEX_DIR)

        logger.info("Saved FAISS index + texts.pkl + meta.pkl to %s", INDEX_DIR)
        print("DONE: FAISS index created at:", INDEX_DIR)

    except Exception as e:
        logger.exception("Failed saving index files: %s", e)
        try:
            if TEMP_INDEX_DIR.exists():
                shutil.rmtree(TEMP_INDEX_DIR)
        except Exception:
            pass
        return


if __name__ == "__main__":
    main()
