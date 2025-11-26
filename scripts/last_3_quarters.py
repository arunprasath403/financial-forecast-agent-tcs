#!/usr/bin/env python3
"""
scripts/last_3_quarters.py
Select last 3 factsheets + last 3 transcripts and write data/selected_pdfs.json
"""
import re
import json
import logging
from pathlib import Path
from urllib.parse import unquote
from pdfminer.high_level import extract_text

# logging
log = logging.getLogger("last_3_quarters")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# repo-root resolver
def find_repo_root(start: Path = None, markers=("data", ".git")) -> Path:
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(10):
        for m in markers:
            if (cur / m).exists():
                return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = find_repo_root()
PDF_DIR = PROJECT_ROOT / "data" / "docs" / "screener_pdfs"
OUT_JSON = PROJECT_ROOT / "data" / "selected_pdfs.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

log.info("PROJECT_ROOT = %s", PROJECT_ROOT)
log.info("PDF_DIR -> %s (exists=%s)", PDF_DIR, PDF_DIR.exists())
log.info("OUT_JSON -> %s (exists=%s)", OUT_JSON, OUT_JSON.exists())

# constants
PAGES_TO_READ = 3

TRANSCRIPT_PAT = re.compile(
    r"(transcript|investor call|earnings call|conference call|q&a|q & a|operator:|participants|scrip code|symbol\s*-\s*tcs)",
    re.I
)
FACTSHEET_PAT = re.compile(r"(fact\s*sheet|financial results|quarterly results|consolidated results)", re.I)
YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})", re.I)
QUARTER_PATTERN = re.compile(r"Q([1-4])", re.I)
DATE_PATTERNS = [
    re.compile(r"(?:held on|on)\s+([A-Za-z]{3,9})\s+(\d{1,2}),\s*(20\d{2})", re.I),
    re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2}),\s*(20\d{2})", re.I),
    re.compile(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})", re.I)
]
MONTH_TO_Q = {
    "jan":1,"feb":1,"mar":1,
    "apr":2,"may":2,"jun":2,
    "jul":3,"aug":3,"sep":3,
    "oct":4,"nov":4,"dec":4
}

# helpers
def read_text(path: Path):
    try:
        return extract_text(str(path), maxpages=PAGES_TO_READ) or ""
    except Exception as e:
        log.warning("extract_text failed for %s: %s", path, e)
        return ""

def detect_type(text, filename):
    fname = unquote(filename).lower()
    # prefer filename markers first (helps with scanned PDFs)
    if FACTSHEET_PAT.search(fname) or "fact" in fname:
        return "factsheet"
    if TRANSCRIPT_PAT.search(fname) or "transcript" in fname:
        return "transcript"
    if FACTSHEET_PAT.search(text):
        return "factsheet"
    if TRANSCRIPT_PAT.search(text):
        return "transcript"
    return None

def detect_year(text, filename):
    for p in DATE_PATTERNS:
        m = p.search(text)
        if m:
            try:
                return int(m.groups()[-1])
            except Exception:
                pass
    m = YEAR_PATTERN.search(filename)
    if m:
        return int(m.group(1))
    return None

def detect_quarter(text, filename):
    mq = QUARTER_PATTERN.search(filename)
    if mq:
        try:
            return int(mq.group(1))
        except Exception:
            pass
    low = (text or "").lower()
    for mon, q in MONTH_TO_Q.items():
        if mon in low:
            return q
    return None

# main
def main():
    docs = []

    if not PDF_DIR.exists():
        log.error("PDF_DIR does not exist: %s", PDF_DIR)
        return

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        txt = read_text(pdf)
        log.debug("Scanning %s (text_len=%d)", pdf.name, len(txt))
        typ = detect_type(txt, pdf.name)
        if not typ:
            log.debug("Skipped %s (no type)", pdf.name)
            continue

        year = detect_year(txt, pdf.name)
        quarter = detect_quarter(txt, pdf.name)
        if not year:
            log.debug("Skipped %s (no year)", pdf.name)
            continue

        docs.append({
            "path": str(pdf.resolve()),
            "name": pdf.name,
            "type": typ,
            "qy": [year, quarter]
        })

    facts = [d for d in docs if d["type"] == "factsheet"]
    trans = [d for d in docs if d["type"] == "transcript"]

    facts.sort(key=lambda d: (d["qy"][0], d["qy"][1] or 0), reverse=True)
    trans.sort(key=lambda d: (d["qy"][0], d["qy"][1] or 0), reverse=True)

    selected = facts[:3] + trans[:3]

    OUT_JSON.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print(f"FACTSHEETS FOUND: {len(facts)}")
    print(f"TRANSCRIPTS FOUND: {len(trans)}\n")
    print("SELECTED FILES:\n")
    for s in selected:
        print(s)

if __name__ == "__main__":
    main()
