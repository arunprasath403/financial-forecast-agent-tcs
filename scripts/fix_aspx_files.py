#!/usr/bin/env python3
"""
scripts/download_pdfs.py
Robust downloader + fixer for screener PDFs
"""
import os
import sys
import re
import time
import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse
import mimetypes

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------
# Repo-safe paths
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
BASE_DIR = REPO_ROOT / "data" / "docs" / "screener_pdfs"
LINKS_FILE = REPO_ROOT / "data" / "docs" / "screener_links.txt"

# constants
RETRY_DELS = ["?Pname=", "?id=", "?"]  # common query markers for AnnPdfOpen.aspx

# -----------------------
# Logging
# -----------------------
log = logging.getLogger("download_pdfs")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# -----------------------
# HTTP session with retries
# -----------------------
def build_session(retries=3, backoff=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; screener-downloader/1.0)"})
    return s

SESSION = build_session()

# -----------------------
# Utilities
# -----------------------
def is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head.startswith(b"%PDF-")
    except Exception:
        return False

def safe_name_from_url(url: str) -> str:
    base = os.path.basename(urlparse(url).path) or "doc"
    base = base.split("?")[0]
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{h}_{base}"

def unique_target(path: Path) -> Path:
    """
    If `path` exists, return a new Path with suffix (1),(2),... inserted before extension.
    """
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def try_rename_pdf_file(p: Path):
    """
    If file content is PDF but extension not .pdf, rename safely.
    Returns new Path or None if no rename done.
    """
    if p.suffix.lower() == ".pdf":
        return None
    try:
        if not p.exists():
            return None
        if is_pdf(p):
            candidate = p.with_suffix(".pdf")
            candidate = unique_target(candidate)
            p.rename(candidate)
            return candidate
    except Exception as e:
        log.exception("Error renaming %s : %s", p, e)
    return None

# -----------------------
# Downloading
# -----------------------
def download_and_save(url: str, out_path: Path, session: requests.Session = SESSION, timeout=30):
    """
    Downloads `url` into same dir as out_path but chooses filename from content-disposition or url.
    Returns (final_path_or_None, status_code_or_None, content_type_or_error_str)
    """
    try:
        r = session.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()

        # try content-disposition filename
        cd = r.headers.get("content-disposition", "")
        fname = None
        if "filename=" in cd:
            # naive parse
            m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', cd)
            if m:
                fname = m.group(1).strip().strip('";\'')

        # fallback to url basename
        if not fname:
            fname = os.path.basename(urlparse(url).path) or safe_name_from_url(url)

        # decide extension
        ext = None
        if ctype == "application/pdf":
            ext = ".pdf"
        elif fname and "." in fname:
            ext = "." + fname.split(".")[-1]
        else:
            ext = mimetypes.guess_extension(ctype) or ".pdf"

        # prepare output filename in same folder as out_path
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        final_name = f"{Path(fname).stem}{ext}"
        target = out_dir / final_name
        if target.exists():
            target = unique_target(target)

        # write atomically
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        tmp.replace(target)

        # verify magic bytes if content-type unknown
        if not target.suffix.lower() == ".pdf":
            try:
                if target.read_bytes()[:5].startswith(b"%PDF-"):
                    # rename to .pdf if needed
                    new_target = target.with_suffix(".pdf")
                    new_target = unique_target(new_target)
                    target.rename(new_target)
                    target = new_target
            except Exception:
                pass

        return target, r.status_code, ctype

    except Exception as e:
        return None, None, str(e)

# -----------------------
# links map loader
# -----------------------
def load_links_map():
    m = {}
    if not LINKS_FILE.exists():
        return m
    try:
        with LINKS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                key = os.path.basename(urlparse(url).path).split("?")[0]
                if key:
                    m.setdefault(key, []).append(url)
    except Exception as e:
        log.exception("Failed to load links map: %s", e)
    return m

# -----------------------
# Main logic
# -----------------------
def main():
    if not BASE_DIR.exists():
        log.error("Directory not found: %s", BASE_DIR)
        sys.exit(1)

    links_map = load_links_map()
    renamed = []
    re_downloaded = []
    left = []

    for p in sorted(BASE_DIR.iterdir()):
        if p.is_dir():
            continue
        name = p.name
        log.info("Checking: %s", name)

        # If it's already a PDF (magic bytes), ensure it has .pdf extension
        if is_pdf(p):
            if not name.lower().endswith(".pdf"):
                newp = try_rename_pdf_file(p)
                log.info(" Renamed PDF: %s -> %s", p, newp)
                renamed.append((p, newp))
            else:
                log.info(" Already PDF - ok")
            continue

        # Not a PDF - try to find original URL via links_map using basename
        base = os.path.basename(name).split("?")[0]
        candidates = links_map.get(base, []) or []
        success = False
        if candidates:
            for url in candidates:
                log.info("  Attempt re-download from URL: %s", url)
                out, status, info = download_and_save(url, p)
                if out and out.exists():
                    log.info("  Downloaded -> %s (status=%s, ctype=%s)", out, status, info)
                    re_downloaded.append((p, url, out))
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    success = True
                    break
                else:
                    log.warning("  Download failed: %s (url=%s)", info, url)
            if success:
                continue

        # Try to extract embedded PDF url inside file (HTML captured 저장)
        try:
            txt = p.read_text(errors="ignore")
            m = re.search(r"(https?://[^\s'\"<>]+\.pdf[^\s'\"<>]*)", txt, flags=re.I)
            if m:
                url = m.group(1)
                log.info("  Found embedded pdf URL in file: %s", url)
                out, status, info = download_and_save(url, p)
                if out and out.exists():
                    log.info("  Downloaded -> %s (status=%s, ctype=%s)", out, status, info)
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    re_downloaded.append((p, url, out))
                    continue
        except Exception:
            pass

        # Last resort: try to re-download by attempting to add typical query markers (best-effort)
        tried_marker = False
        for q in RETRY_DELS:
            tried_marker = True
            guess = str(p.name) + q
            # attempt reconstructing a likely URL by searching links_map keys containing the p.name
            for key, urls in links_map.items():
                if p.name in key or key in p.name:
                    for url in urls:
                        log.info("  Trying candidate url (marker fallback): %s", url)
                        out, status, info = download_and_save(url, p)
                        if out and out.exists():
                            log.info("  Downloaded -> %s (status=%s, ctype=%s)", out, status, info)
                            re_downloaded.append((p, url, out))
                            try:
                                p.unlink()
                            except Exception:
                                pass
                            success = True
                            break
                    if success:
                        break
            if success:
                break

        if success:
            continue

        log.warning("  Left as-is for manual inspection: %s", p)
        left.append(p)

    # summary (print/log)
    log.info("Summary: Renamed=%d, Re-downloaded=%d, Left=%d", len(renamed), len(re_downloaded), len(left))
    if left:
        log.info("Manual files (first 10): %s", [str(x) for x in left[:10]])

    # Also return for programmatic use
    return {"renamed": renamed, "re_downloaded": re_downloaded, "left": left}

if __name__ == "__main__":
    main()
