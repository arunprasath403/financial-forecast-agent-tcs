#!/usr/bin/env python3
import re
import json
from pathlib import Path
from urllib.parse import unquote
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

# Get project root (adjust parents[] depending where this file is)
_log = logging.getLogger("tcs_agent.financial_extractor")
if not _log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)

# project root (adjust parents if file location differs)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# default (project-relative)
default_out = PROJECT_ROOT / "data" / "selected_pdfs.json"

# absolute path you provided (Windows)
provided_abs = Path(r"C:\Users\arun.prasathr\financial-forecast-agent-tcs\data\selected_pdfs.json")

# allow override via env var SELECTED_PDFS
env_path = os.environ.get("SELECTED_PDFS")

if env_path:
    OUT_JSON = Path(env_path).resolve()
elif provided_abs.exists():
    OUT_JSON = provided_abs.resolve()
else:
    OUT_JSON = default_out.resolve()

# ensure parent folder exists
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

print("Using OUT_JSON =", OUT_JSON)

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

# date detection from content
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


# --- HELPERS ---
def read_text(path: Path):
    try:
        return extract_text(str(path), maxpages=PAGES_TO_READ) or ""
    except:
        return ""


def detect_type(text, filename):
    fname = unquote(filename).lower()

    if FACTSHEET_PAT.search(fname) or FACTSHEET_PAT.search(text):
        return "factsheet"

    if TRANSCRIPT_PAT.search(fname) or TRANSCRIPT_PAT.search(text):
        return "transcript"

    return None


def detect_year(text, filename):
    # 1. Try date inside transcript
    for p in DATE_PATTERNS:
        m = p.search(text)
        if m:
            return int(m.groups()[-1])

    # 2. Try filename
    m = YEAR_PATTERN.search(filename)
    if m:
        return int(m.group(1))

    return None


def detect_quarter(text, filename):
    # from filename
    mq = QUARTER_PATTERN.search(filename)
    if mq:
        return int(mq.group(1))

    # from month text
    low = text.lower()
    for mon, q in MONTH_TO_Q.items():
        if mon in low:
            return q

    return None


# --- MAIN ---
def main():
    docs = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        txt = read_text(pdf)
        typ = detect_type(txt, pdf.name)
        if not typ:
            continue

        year = detect_year(txt, pdf.name)
        quarter = detect_quarter(txt, pdf.name)
        if not year:
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

    # sort newest first
    facts.sort(key=lambda d: (d["qy"][0], d["qy"][1] or 0), reverse=True)
    trans.sort(key=lambda d: (d["qy"][0], d["qy"][1] or 0), reverse=True)

    # select last 3 each
    selected = facts[:3] + trans[:3]

    OUT_JSON.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print(f"FACTSHEETS FOUND: {len(facts)}")
    print(f"TRANSCRIPTS FOUND: {len(trans)}\n")
    print("SELECTED FILES:\n")
    for s in selected:
        print(s)

if __name__ == "__main__":
    main()
