#!/usr/bin/env python3
import re
import json
import logging
from pathlib import Path
from urllib.parse import unquote
from pdfminer.high_level import extract_text
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Logging
# -------------------------
log = logging.getLogger("tcs_agent.financial_extractor")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# ------------------------------------------------------------------
# Project-root-based paths (portable)
# ------------------------------------------------------------------
# If this file lives in repo/app/tools/..., parents[2] points to the repo root.
# If this file is somewhere else, adjust parents[n] accordingly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data locations (built from project root)
PDF_DIR = PROJECT_ROOT / "data" / "docs" / "screener_pdfs"
OUT_JSON = PROJECT_ROOT / "data" / "selected_pdfs.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

log.info("PROJECT_ROOT = %s", PROJECT_ROOT)
log.info("PDF_DIR = %s (exists=%s)", PDF_DIR, PDF_DIR.exists())
log.info("OUT_JSON = %s", OUT_JSON)

# Other constants
PAGES_TO_READ = 3

# --- REGEX ---
TRANSCRIPT_PAT = re.compile(
    r"(transcript|investor call|earnings call|conference call|q&a|q & a|operator:|participants|scrip code|symbol\s*-\s*tcs)",
    re.I
)

FACTSHEET_PAT = re.compile(r"(fact\s*sheet|financial results|quarterly results|consolidated results)", re.I)

YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})", re.I)
QUARTER_PATTERN = re.compile(r"Q([1-4])", re.I)

# date detection from content (multiple formats)
DATE_PATTERNS = [
    re.compile(r"(?:held on|on)\s+([A-Za-z]{3,9})\s+(\d{1,2}),\s*(20\d{2})", re.I),
    re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2}),\s*(20\d{2})", re.I),
    re.compile(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})", re.I),
]

MONTH_TO_Q = {
    "jan": 1, "january": 1, "feb": 1, "february": 1, "mar": 1, "march": 1,
    "apr": 2, "april": 2, "may": 2, "jun": 2, "june": 2,
    "jul": 3, "july": 3, "aug": 3, "august": 3, "sep": 3, "september": 3,
    "oct": 4, "october": 4, "nov": 4, "november": 4, "dec": 4, "december": 4,
}


# --- HELPERS ---
def read_text(path: Path) -> str:
    try:
        return extract_text(str(path), maxpages=PAGES_TO_READ) or ""
    except Exception as e:
        log.warning("Failed to extract text from %s: %s", path, e, exc_info=False)
        return ""


def detect_type(text: str, filename: str):
    fname = unquote(filename).lower()
    if FACTSHEET_PAT.search(fname) or FACTSHEET_PAT.search(text):
        return "factsheet"
    if TRANSCRIPT_PAT.search(fname) or TRANSCRIPT_PAT.search(text):
        return "transcript"
    return None


def detect_year(text: str, filename: str):
    # 1. Try date inside transcript
    for p in DATE_PATTERNS:
        m = p.search(text)
        if m:
            try:
                return int(m.groups()[-1])
            except Exception:
                pass

    # 2. Try filename
    m = YEAR_PATTERN.search(filename)
    if m:
        return int(m.group(1))
    return None


def detect_quarter(text: str, filename: str):
    # from filename like Q1/Q2
    mq = QUARTER_PATTERN.search(filename)
    if mq:
        try:
            return int(mq.group(1))
        except Exception:
            pass

    # from month text
    low = (text or "").lower()
    for mon, q in MONTH_TO_Q.items():
        if mon in low:
            return q

    return None


# --- MAIN ---
def main():
    if not PDF_DIR.exists():
        log.warning("PDF_DIR does not exist: %s", PDF_DIR)
    docs = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        txt = read_text(pdf)
        typ = detect_type(txt, pdf.name)
        if not typ:
            continue

        year = detect_year(txt, pdf.name)
        quarter = detect_quarter(txt, pdf.name)

        if not year:
            # skip files with no year detected
            log.debug("Skipping %s - no year found", pdf.name)
            continue

        docs.append({
            "path": str(pdf.resolve()),
            "name": pdf.name,
            "type": typ,
            "qy": [year, quarter]
        })

    # split
    facts = [d for d in docs if d["type"] == "factsheet"]
    trans = [d for d in docs if d["type"] == "transcript"]

    # sort newest first. handle None quarter by using 0
    facts.sort(key=lambda d: (d["qy"][0], d["qy"][1] or 0), reverse=True)
    trans.sort(key=lambda d: (d["qy"][0], d["qy"][1] or 0), reverse=True)

    # select last 3 each
    selected = facts[:3] + trans[:3]

    # write atomically (safe)
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)  # atomic on most OSes

    print(f"FACTSHEETS FOUND: {len(facts)}")
    print(f"TRANSCRIPTS FOUND: {len(trans)}\n")
    print("SELECTED FILES:\n")
    for s in selected:
        print(s)


if __name__ == "__main__":
    main()
