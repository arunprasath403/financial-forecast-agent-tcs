# app/tools/qualitative_tool.py
"""
QualitativeAnalysisTool (Transcript RAG)
Reads selected_pdfs.json automatically and picks only transcript PDFs.
"""

import os
import re
import json
import logging
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

from app.config import settings
from app.logger import logger

# ---- suppress noisy pdfminer logs ----
for noisy in ["pdfminer", "pdfminer.pdfinterp", "pdfminer.converter"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# optional dependencies
try:
    import pdfplumber
except:
    pdfplumber = None

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except:
    SentimentIntensityAnalyzer = None

try:
    from transformers import pipeline as transformers_pipeline
except:
    transformers_pipeline = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except:
    SentenceTransformer = None
    st_util = None

# -------------------------------------------------------------------
# PATH to selected_pdfs.json (YOUR FILE)
# -------------------------------------------------------------------
SELECTED_PDFS = Path(r"D:\Forecast_agent_TCS\data\selected_pdfs.json")


# -------------------- LOAD TRANSCRIPT LIST -------------------------
def _load_transcript_paths_from_json() -> List[str]:
    """Return list of transcript PDF paths from selected_pdfs.json."""
    if not SELECTED_PDFS.exists():
        logger.error("selected_pdfs.json not found at %s", SELECTED_PDFS)
        return []

    try:
        data = json.load(open(SELECTED_PDFS, "r", encoding="utf-8"))
    except Exception as e:
        logger.exception("Failed to read selected_pdfs.json: %s", e)
        return []

    transcript_paths = []
    for item in data:
        if str(item.get("type", "")).lower() == "transcript":
            p = item.get("path")
            if p and os.path.exists(p):
                transcript_paths.append(p)
            else:
                logger.warning("Transcript path missing or not found: %s", p)

    logger.info("Loaded %d transcript PDFs from selected_pdfs.json", len(transcript_paths))
    return transcript_paths


# -------------------- LOAD PDF TEXT -------------------------------
def _load_pdf_text(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    if not os.path.exists(path):
        return ""

    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception:
        logger.exception("pdfplumber failed: %s", path)
        return ""


# -------------------- EMBEDDING UTILS ------------------------------
def _load_model():
    """
    Try SBERT → else try TF-IDF → else tiny hashing fallback.
    Always returns an object with .encode(texts) → np.ndarray
    """
    # --- Try SBERT (SentenceTransformer) ---
    if SentenceTransformer is not None:
        try:
            import torch
            # force CPU device (avoid meta tensor failure)
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

            class _SbertWrapper:
                def __init__(self, m):
                    self.m = m

                def encode(self, texts, **kwargs):
                    return self.m.encode(
                        texts,
                        convert_to_tensor=False,   # IMPORTANT: keep as numpy instead of torch.Tensor
                        show_progress_bar=False
                    )

            logger.info("Loaded SBERT model successfully on CPU")
            return _SbertWrapper(model)

        except Exception as e:
            logger.error(f"SBERT failed to load: {e}. Falling back to TF-IDF.")

    # --- TF-IDF Fallback ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        class _TfidfWrapper:
            def __init__(self):
                self.v = TfidfVectorizer(max_features=2048, stop_words="english")

            def encode(self, texts, **kwargs):
                arr = self.v.fit_transform(texts).toarray().astype("float32")
                return arr

        logger.info("Using TF-IDF embedding fallback.")
        return _TfidfWrapper()

    except Exception as e2:
        logger.error(f"TF-IDF fallback failed: {e2}. Using tiny hashing fallback.")

    # --- Tiny Hashing Fallback ---
    import numpy as np

    class _TinyHash:
        def encode(self, texts, **kwargs):
            outs = []
            for t in texts:
                h = np.frombuffer(t.encode("utf8")[:1024].ljust(1024, b'\0'),
                                  dtype=np.uint8).astype("float32")
                outs.append(h)
            return np.vstack(outs)

    logger.info("Using tiny hashing fallback.")
    return _TinyHash()


def _embed_docs(model, docs: List[str]):
    if not model:
        return None, None, None

    chunks = []
    doc_map = []

    for i, d in enumerate(docs):
        parts = [p.strip() for p in re.split(r"\n{1,}", d) if len(p.strip()) > 10]
        for p in parts:
            chunks.append(p)
            doc_map.append(i)

    if not chunks:
        return None, None, None

    emb = model.encode(chunks, convert_to_tensor=True)
    return chunks, doc_map, emb


# -------------------- SENTIMENT UTILS ------------------------------
def _get_sentiment_engine():
    if transformers_pipeline:
        try:
            return transformers_pipeline("sentiment-analysis")
        except:
            pass
    if SentimentIntensityAnalyzer:
        try:
            return SentimentIntensityAnalyzer()
        except:
            pass
    return None


def _sentiment(engine, text: str):
    if not engine:
        return {"label": "unknown", "score": None}

    try:
        # transformers
        if hasattr(engine, "__call__") and hasattr(engine, "tokenizer"):
            out = engine(text[:512])
            if out:
                o = out[0]
                return {"label": o["label"], "score": float(o["score"])}
    except:
        pass

    # nltk fallback
    try:
        scores = engine.polarity_scores(text[:2000])
        comp = scores["compound"]
        label = "neutral"
        if comp >= 0.05:
            label = "positive"
        elif comp <= -0.05:
            label = "negative"
        return {"label": label, "score": comp}
    except:
        return {"label": "unknown", "score": None}


# -------------------- FWD LOOKING DETECTION -------------------------
_FORWARD = [
    r"\bexpect", r"\bwill\b", r"guidance", r"\bplan", r"\bintend",
    r"target", r"outlook", r"forecast", r"aim", r"anticipat"
]


def _find_forward(text: str):
    found = []
    pat = re.compile("|".join(_FORWARD), re.I)
    for m in pat.finditer(text):
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 300)
        snip = text[start:end].strip()
        if len(snip) > 40:
            found.append({"match": m.group(0), "snippet": snip})
    return found


# ===================================================================
#                           MAIN TOOL
# ===================================================================
class QualitativeAnalysisTool:
    def __init__(self):
        logger.info("QualitativeAnalysisTool initialized")
        self.model = _load_model()
        self.sentiment_engine = _get_sentiment_engine()

    def analyze(self, topics=None) -> Dict[str, Any]:
        """Runs full qualitative analysis using transcripts from selected_pdfs.json"""

        transcript_paths = _load_transcript_paths_from_json()
        if not transcript_paths:
            return {"error": "No transcript PDFs found in selected_pdfs.json"}

        # load pdf texts
        docs = []
        file_names = []
        for p in transcript_paths:
            txt = _load_pdf_text(p)
            if txt:
                docs.append(txt)
                file_names.append(os.path.basename(p))
            else:
                logger.warning("No text extracted: %s", p)

        if not docs:
            return {"error": "No usable transcript text extracted"}

        # embed
        chunks, doc_map, embeds = _embed_docs(self.model, docs)

        # default topics
        if not topics:
            topics = ["demand", "attrition", "guidance", "margins", "deal wins"]

        # topic search
        topic_hits = {}
        if self.model and embeds is not None:
            for topic in topics:
                q = self.model.encode([topic], convert_to_tensor=True)
                hits = st_util.semantic_search(q, embeds, top_k=5)[0]
                enriched = []
                for h in hits:
                    cid = h["corpus_id"]
                    enriched.append({
                        "file": file_names[doc_map[cid]],
                        "score": float(h["score"]),
                        "snippet": chunks[cid]
                    })
                topic_hits[topic] = enriched
        else:
            # substring fallback
            for topic in topics:
                result = []
                for i, doc in enumerate(docs):
                    for m in re.finditer(topic, doc, re.I):
                        sn = doc[max(0, m.start()-40):m.end()+80]
                        result.append({"file": file_names[i], "score": 1.0, "snippet": sn})
                        if len(result) >= 5:
                            break
                topic_hits[topic] = result

        # sentiment analysis
        sentiments = []
        for i, doc in enumerate(docs):
            sentiments.append({
                "file": file_names[i],
                "sentiment": _sentiment(self.sentiment_engine, doc[:3000])
            })

        # forward looking statements
        forward = []
        for i, doc in enumerate(docs):
            for f in _find_forward(doc):
                forward.append({
                    "file": file_names[i],
                    "match": f["match"],
                    "snippet": f["snippet"]
                })

        # themes (simple frequent words)
        words = Counter()
        for d in docs:
            toks = re.findall(r"\b[a-zA-Z]{4,}\b", d.lower())
            words.update(toks)
        common = words.most_common(15)

        return {
            "files_used": file_names,
            "themes": common,
            "topic_hits": topic_hits,
            "sentiment": sentiments,
            "forward_statements": forward
        }


# -------------------- CLI TEST -------------------------
if __name__ == "__main__":
    tool = QualitativeAnalysisTool()
    res = tool.analyze()
    print(json.dumps(res, indent=2, ensure_ascii=False))
