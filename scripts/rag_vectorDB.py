# scripts/step2_ingest_rag_fixed.py
import os
import sys
import pickle
from pathlib import Path

sys.path.append(os.getcwd())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.logger import logger

VECTOR_ROOT = Path(settings.VECTOR_DIR)
INDEX_DIR = VECTOR_ROOT / "faiss_index"


# -----------------------------------------------------------
# LOAD FROM selected_pdfs.json ONLY
# -----------------------------------------------------------
def load_selected_pdfs():
    selected_file = Path("data/selected_pdfs.json")
    if not selected_file.exists():
        logger.error("selected_pdfs.json NOT FOUND at data/selected_pdfs.json")
        return []

    import json
    data = json.loads(selected_file.read_text())

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


# -----------------------------------------------------------
# Simple page-level transcript detector
# -----------------------------------------------------------
TRANSCRIPT_MARKERS = [
    "operator", "question and answer", "q&a", "earnings conference call",
    "earnings call", "moderator:", "thank you", "open for questions",
    "tata consultancy services earnings conference call", "investor call",
    "investor conference", "participants", "question from", "panelist"
]


def is_transcript_text(txt: str) -> bool:
    """
    Heuristic: returns True if the page text looks like a transcript.
    - checks for known transcript markers
    - checks for speaker-colon patterns (e.g. 'Moderator:' / 'John Doe:')
    - short heuristic threshold to avoid false positives
    """
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
        logger.error("No PDFs to ingest.")
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

        logger.info("Loading PDF: %s", pdf_path)

        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()

            # classify each page as transcript/factsheet/other using page text
            for d in pages:
                # ensure metadata
                if d.metadata is None:
                    d.metadata = {}

                page_text = d.page_content if hasattr(d, "page_content") else (d.content if hasattr(d, "content") else "")
                # set page_type based on heuristic
                d.metadata["page_type"] = "transcript" if is_transcript_text(page_text) else None

            # Add metadata to each document and prefer page_type over file-level type
            for d in pages:
                d.metadata = d.metadata or {}

                d.metadata["file_name"] = pdf_name
                d.metadata["source"] = pdf_path
                # prefer page-level detection, otherwise fall back to JSON-provided type
                d.metadata["type"] = d.metadata.get("page_type") or pdf_type or "unknown"
                d.metadata["year"] = year
                d.metadata["quarter"] = quarter

            docs.extend(pages)

        except Exception as e:
            logger.exception("Failed loading: %s : %s", pdf_path, e)

    if not docs:
        logger.error("No documents extracted!")
        return

    # ---------------------------------------------
    # Chunking
    # ---------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    # ---------------------------------------------
    # Embed + FAISS
    # ---------------------------------------------
    embed_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectordb = FAISS.from_documents(chunks, embed_model)
    vectordb.save_local(str(INDEX_DIR))

    # ---------------------------------------------
    # Save aligned texts + metadata
    # ---------------------------------------------
    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]

    with open(INDEX_DIR / "texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    with open(INDEX_DIR / "meta.pkl", "wb") as f:
        pickle.dump(metas, f)

    logger.info("Saved FAISS index + texts.pkl + meta.pkl to %s", INDEX_DIR)
    print("DONE: FAISS index created at:", INDEX_DIR)


if __name__ == "__main__":
    main()
