# app/tools/financial_extractor.py
"""
FinancialDataExtractorTool (updated)
-----------------------------------

Notes:
- Groups FAISS chunks by file_name so each fact-sheet is processed once.
- Accepts meta keys: 'type', 'doc_type', 'category' to detect factsheets.
- Returns both parsed numeric value and the raw matched string/unit.
"""

import re
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable
from collections import defaultdict

from app.config import settings
from app.logger import logger

# optional dependencies / helpers
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from app.tools.pdf_extractor import extract_text_from_pdf
except Exception:
    extract_text_from_pdf = None


INDEX_DIR = Path(settings.VECTOR_DIR) / "faiss_index"
TEXTS_FILE = INDEX_DIR / "texts.pkl"
META_FILE = INDEX_DIR / "meta.pkl"


def _clean_number(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    s = s.replace("₹", "").replace("Rs.", "").replace("INR", "").replace("$", "")
    # preserve unit tokens (Mn, bn, %), but remove other junk for numeric parse
    unit = None
    unit_match = re.search(r"\b(mn|m|million|bn|billion|bn|%)\b", s, flags=re.I)
    if unit_match:
        unit = unit_match.group(0).lower()
    # remove everything except digits, dot, minus, parentheses and commas
    cleaned = re.sub(r"[^\d\.\-\(\),eE]", " ", s)
    cleaned = cleaned.replace(",", "").strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    cleaned = cleaned.replace("%", "")
    try:
        val = float(cleaned)
        return val
    except Exception:
        return None


def _find_number_patterns_in_texts(patterns: List[str], texts: List[str]):
    results = []
    for idx, t in enumerate(texts):
        for pat in patterns:
            for m in re.finditer(pat, t, flags=re.I | re.M):
                groups = m.groups()
                val = None
                raw_match = m.group(0)
                unit = None
                if groups:
                    for g in groups:
                        if not g:
                            continue
                        num = _clean_number(g)
                        if num is not None:
                            val = num
                            # capture unit token if present in group
                            um = re.search(r"\b(mn|m|million|bn|billion|%)\b", str(g), flags=re.I)
                            unit = um.group(0).lower() if um else None
                            break
                if val is None:
                    mm = re.search(r"[\d\(\)\,\.]{2,}", m.group(0))
                    if mm:
                        val = _clean_number(mm.group(0))
                snippet = m.group(0)[:400].replace("\n", " ")
                if val is not None:
                    results.append((val, idx, snippet, pat, raw_match, unit))
    return results


def load_faiss_texts_and_metas():
    if not TEXTS_FILE.exists():
        logger.warning("FAISS texts.pkl not found at %s", TEXTS_FILE)
        return [], []
    try:
        texts = pickle.load(open(TEXTS_FILE, "rb"))
    except Exception as e:
        logger.exception("Failed to load texts.pkl: %s", e)
        texts = []

    if META_FILE.exists():
        try:
            metas = pickle.load(open(META_FILE, "rb"))
        except Exception as e:
            logger.exception("Failed to load meta.pkl: %s", e)
            metas = []
    else:
        metas = []
        logger.warning("meta.pkl missing — using dummy metadata")

    # ensure metas length >= texts length by padding with simple dicts
    if len(metas) < len(texts):
        metas = metas + [{"source": "unknown", "file_name": None} for _ in range(len(texts) - len(metas))]
    return texts, metas


def load_pdf_text(pdf_path: str) -> str:
    # optionally suppress noisy pdfminer logs in the process that imports pdfplumber
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    if extract_text_from_pdf:
        try:
            txt = extract_text_from_pdf(pdf_path)
            if txt and len(txt) > 100:
                return txt
        except Exception:
            logger.exception("extract_text_from_pdf failed for %s", pdf_path)
    if pdfplumber:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except Exception:
            logger.exception("pdfplumber failed for %s", pdf_path)
    try:
        with open(pdf_path, "rb") as f:
            f.read(16)  # just ensure file exists
        return ""
    except Exception:
        return ""


def _get_meta_doc_type(meta: dict) -> str:
    # look for multiple possible keys
    for k in ("type", "doc_type", "category"):
        v = meta.get(k)
        if v:
            return str(v).lower()
    return ""


def _group_texts_by_file(texts: List[str], metas: List[dict]) -> Tuple[List[str], List[dict]]:
    """Group chunks by file_name; return joined texts list and representative metas list"""
    groups = defaultdict(list)   # file_name -> list of text chunks
    meta_by_file = {}            # file_name -> meta (first seen)
    for idx, (t, m) in enumerate(zip(texts, metas)):
        fname = (m.get("file_name") or os.path.basename(str(m.get("source") or f"chunk_{idx}"))).strip()
        if not fname:
            fname = f"chunk_{idx}"
        groups[fname].append(t)
        if fname not in meta_by_file:
            meta_by_file[fname] = m
    joined_texts = []
    joined_metas = []
    for fname, chunks in groups.items():
        joined_texts.append("\n".join(chunks))
        meta = meta_by_file.get(fname, {"file_name": fname, "source": fname})
        joined_metas.append(meta)
    return joined_texts, joined_metas


def extract_financial_metrics(pdf_path: str = None) -> Dict[str, Any]:
    logger.info("Extracting financial metrics from: %s", pdf_path if pdf_path else "<faiss-texts-fallback>")

    texts = []
    metas = []
    sources_used = set()

    if pdf_path:
        t = load_pdf_text(pdf_path)
        if not t:
            return {"error": "no_text_extracted_from_pdf", "source_file": os.path.basename(pdf_path)}
        texts = [t]
        metas = [{"source": pdf_path, "file_name": os.path.basename(pdf_path), "type": "factsheet"}]
    else:
        texts_all, metas_all = load_faiss_texts_and_metas()
        if not texts_all:
            return {"error": "no_faiss_texts_found"}
        # filter only factsheet chunks (multi-key support)
        filtered_pairs = []
        for i, m in enumerate(metas_all):
            doc_type = _get_meta_doc_type(m)
            fname = (m.get("file_name") or os.path.basename(str(m.get("source") or f"chunk_{i}")) or "").lower()
            if doc_type == "factsheet" or doc_type == "fact_sheet" or doc_type == "fact sheet":
                filtered_pairs.append((i, m))
            elif "fact" in fname:
                filtered_pairs.append((i, m))
        if not filtered_pairs:
            # fallback: include all texts (very last resort)
            indices = list(range(len(texts_all)))
            filtered_texts = texts_all
            filtered_metas = metas_all
        else:
            indices = [i for i, _ in filtered_pairs]
            filtered_texts = [texts_all[i] for i in indices]
            filtered_metas = [metas_all[i] for i in indices]

        # group by file_name to get one joined text per file
        texts, metas = _group_texts_by_file(filtered_texts, filtered_metas)

    if not texts:
        return {"error": "no_factsheet_texts_found"}

    for m in metas:
        fn = (m.get("file_name") or os.path.basename(str(m.get("source") or "unknown"))).strip()
        sources_used.add(fn)

    # patterns (same as your current ones)
    revenue_patterns = [
        r"Revenue(?:\s+from\s+Operations)?[^0-9A-Za-z]*?([₹\$]?\s?[\d\.,]{3,}\s*(?:Mn|M|million|billion|bn)?)",
        r"(?:Revenue|Total Revenue|Net Income at)\s*(?:at|:)?\s*([₹\$]?\s?[\d\.,]{3,}\s*(?:Mn|M|million|billion|bn)?)",
        r"Revenue[^0-9]*([\d,]{3,}\.?\d*)",
        r"([\d\.,]{3,}\s*(?:Mn|M|million|billion|bn))\s+(?:Revenue|Revenue\s+`?)",
        r"constant currency revenue (?:up|down)?\s*([\d\.\-]+)\s*%?",
        r"up\s+([\d\.\-]+)\s*%.*revenue",
    ]
    netprofit_patterns = [
        r"Net\s+Income(?:\s+at)?\s*[₹\$]?\s*([,\d\.\s]{2,}?\d)\b",
        r"Net\s+Profit[^0-9A-Za-z]*?([₹\$]?\s?[\d\.,]{3,}\s*(?:Mn|M|million|billion|bn)?)",
        r"Profit\s+After\s+Tax[^0-9A-Za-z]*?([₹\$]?\s?[\d\.,]{3,}\s*(?:Mn|M|million|billion|bn)?)",
        r"\bPAT\b[^0-9A-Za-z]*?([₹\$]?\s?[\d\.,]{3,}\s*(?:Mn|M|million|billion|bn)?)",
    ]
    margin_patterns = [
        r"Operating\s+Margin[^0-9]*([\d\.\-]+)\s*%",
        r"Operating\s+Income\s+at\s*[₹\$]?\s*([,\d\.\s]{2,}?\d)\s*,.*Operating\s+Margin\s+of\s+([\d\.\-]+)\s*%",
        r"Operating\s+Income[^0-9]*([₹\$]?\s?[\d\.,]{3,}\s*(?:Mn|M)?)",
        r"EBIT\s+Margin[^0-9]*([\d\.\-]+)\s*%",
    ]
    yoy_patterns = [r"YoY[^0-9]*([\d\.\-]+)\s*%", r"year[-\s]over[-\s]year[^0-9]*([\d\.\-]+)\s*%"]
    qoq_patterns = [r"QoQ[^0-9]*([\d\.\-]+)\s*%", r"quarter[-\s]on[-\s]quarter[^0-9]*([\d\.\-]+)\s*%"]

    revenue_hits = _find_number_patterns_in_texts(revenue_patterns, texts)
    netprofit_hits = _find_number_patterns_in_texts(netprofit_patterns, texts)
    margin_hits = _find_number_patterns_in_texts(margin_patterns, texts)
    yoy_hits = _find_number_patterns_in_texts(yoy_patterns, texts)
    qoq_hits = _find_number_patterns_in_texts(qoq_patterns, texts)

    raw_evidence = []
    chosen_revenue = chosen_net = chosen_margin = chosen_yoy = chosen_qoq = None

    # choose best revenue (largest absolute numeric) but keep raw_match/unit
    if revenue_hits:
        numeric_candidates = [(v, i, s, p, rm, u) for (v, i, s, p, rm, u) in revenue_hits if isinstance(v, (int, float))]
        if numeric_candidates:
            numeric_candidates.sort(key=lambda x: abs(x[0]), reverse=True)
            v, idx, snip, pat, raw_match, unit = numeric_candidates[0]
            chosen_revenue = v
            raw_evidence.append({"type": "revenue", "value": v, "source_idx": idx, "snippet": snip, "pattern": pat, "raw_match": raw_match, "unit": unit})
        else:
            v, idx, snip, pat, raw_match, unit = revenue_hits[0]
            chosen_revenue = v
            raw_evidence.append({"type": "revenue", "value": v, "source_idx": idx, "snippet": snip, "pattern": pat, "raw_match": raw_match, "unit": unit})

    if netprofit_hits:
        numeric_candidates = [(v, i, s, p, rm, u) for (v, i, s, p, rm, u) in netprofit_hits if isinstance(v, (int, float))]
        if numeric_candidates:
            numeric_candidates.sort(key=lambda x: abs(x[0]), reverse=True)
            v, idx, snip, pat, raw_match, unit = numeric_candidates[0]
        else:
            v, idx, snip, pat, raw_match, unit = netprofit_hits[0]
        chosen_net = v
        raw_evidence.append({"type": "net_profit", "value": v, "source_idx": idx, "snippet": snip, "pattern": pat, "raw_match": raw_match, "unit": unit})

    if margin_hits:
        percent_hits = [(v, i, s, p, rm, u) for (v, i, s, p, rm, u) in margin_hits if isinstance(v, (int, float)) and 0 <= abs(v) <= 100]
        if percent_hits:
            # prefer highest absolute percent (but any reasonable heuristic is fine)
            percent_hits.sort(key=lambda x: abs(x[0]), reverse=True)
            v, idx, snip, pat, raw_match, unit = percent_hits[0]
        else:
            v, idx, snip, pat, raw_match, unit = margin_hits[0]
        chosen_margin = v
        raw_evidence.append({"type": "operating_margin", "value": v, "source_idx": idx, "snippet": snip, "pattern": pat, "raw_match": raw_match, "unit": unit})

    if yoy_hits:
        v, idx, snip, pat, raw_match, unit = yoy_hits[0]
        chosen_yoy = v
        raw_evidence.append({"type": "yoy", "value": v, "source_idx": idx, "snippet": snip, "pattern": pat, "raw_match": raw_match, "unit": unit})
    if qoq_hits:
        v, idx, snip, pat, raw_match, unit = qoq_hits[0]
        chosen_qoq = v
        raw_evidence.append({"type": "qoq", "value": v, "source_idx": idx, "snippet": snip, "pattern": pat, "raw_match": raw_match, "unit": unit})

    fact_sources_used = sorted(list(sources_used))
    evidence_texts = []
    for ev in raw_evidence:
        idx = ev.get("source_idx")
        snippet = ev.get("snippet")
        source_fn = metas[idx].get("file_name") if idx is not None and idx < len(metas) else None
        evidence_texts.append(f"{ev['type']} | source={source_fn or 'faiss_chunk_'+str(idx)} | snippet={snippet}")

    result = {
        "total_revenue": chosen_revenue,
        "net_profit": chosen_net,
        "operating_margin": chosen_margin,
        "yoy_growth": chosen_yoy,
        "qoq_growth": chosen_qoq,
        "fact_sources_used": fact_sources_used,
        "evidence_text": "\n\n".join(evidence_texts)[:8000],
        "raw_evidence": raw_evidence,
    }
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(extract_financial_metrics(None), indent=2, ensure_ascii=False))
