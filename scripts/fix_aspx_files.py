# scripts/fix_aspx_files.py
import os
import sys
import requests
from pathlib import Path
import mimetypes
from urllib.parse import urlparse
import hashlib

BASE_DIR = Path("data/docs/screener_pdfs")
LINKS_FILE = Path("data/docs/screener_links.txt")  # optional; used to map back to original URL
RETRY_DELS = ["?Pname=", "?id=", "?"]  # common query markers for AnnPdfOpen.aspx

def is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head.startswith(b"%PDF-")
    except Exception:
        return False

def safe_name_from_url(url: str) -> str:
    # return a safe filename (no collisions) using hash + basename
    base = os.path.basename(urlparse(url).path) or "doc"
    base = base.split("?")[0]
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{h}_{base}"

def try_rename_pdf_file(p: Path):
    # rename file to .pdf if content looks like PDF
    if p.suffix.lower() == ".pdf":
        return None
    candidate = p.with_suffix(".pdf")
    if candidate.exists():
        # avoid clobbering
        candidate = p.with_name(p.stem + "_" + "pdf" + ".pdf")
    p.rename(candidate)
    return candidate

def download_and_save(url: str, out_path: Path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        # try content-disposition
        cd = r.headers.get("content-disposition", "")
        fname = None
        if "filename=" in cd:
            fname = cd.split("filename=")[-1].strip('"; ')
        # fallback to path basename
        if not fname:
            fname = os.path.basename(urlparse(url).path) or None
        # guess extension by content-type
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = None
        if ctype == "application/pdf":
            ext = ".pdf"
        elif fname and "." in fname:
            ext = "." + fname.split(".")[-1]
        else:
            ext = mimetypes.guess_extension(ctype) or ".pdf"

        if not fname:
            fname = safe_name_from_url(url)
        out = out_path.with_name(f"{Path(fname).stem}{ext}")
        # avoid overwriting
        if out.exists():
            out = out.with_name(out.stem + "_" + out.suffix)
        out.write_bytes(r.content)
        return out, r.status_code, ctype
    except Exception as e:
        return None, None, str(e)

def load_links_map():
    # build a map of basename->url for matching poorly named downloaded files (best-effort)
    m = {}
    if not LINKS_FILE.exists():
        return m
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            key = os.path.basename(urlparse(url).path)
            if key:
                m.setdefault(key, []).append(url)
    return m

def main():
    if not BASE_DIR.exists():
        print("Directory not found:", BASE_DIR)
        sys.exit(1)

    links_map = load_links_map()
    renamed = []
    re_downloaded = []
    left = []

    for p in sorted(BASE_DIR.iterdir()):
        if p.is_dir():
            continue
        name = p.name
        lower = name.lower()
        print("Checking:", name)
        if is_pdf(p):
            if not lower.endswith(".pdf"):
                newp = try_rename_pdf_file(p)
                print(" Renamed PDF:", newp)
                renamed.append((p, newp))
            else:
                print(" Already PDF - ok")
            continue

        # not a PDF: try to find original URL by matching basename
        base = os.path.basename(name).split("?")[0]
        candidates = links_map.get(base, []) or []
        if candidates:
            # try each candidate until we get a good PDF
            success = False
            for url in candidates:
                print("  Attempt re-download from URL:", url)
                out, status, info = download_and_save(url, p)
                if out and out.exists():
                    if out.suffix.lower() != ".pdf":
                        # if content-type indicated pdf but extension not set, force .pdf
                        if info == "application/pdf" or out.read_bytes()[:5].startswith(b"%PDF-"):
                            new_out = out.with_suffix(".pdf")
                            out.rename(new_out)
                            out = new_out
                    print("  Downloaded ->", out, "status:", status, "ctype:", info)
                    re_downloaded.append((p, url, out))
                    success = True
                    # optionally delete old file
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    break
                else:
                    print("  Download failed:", info)
            if success:
                continue

        # if no candidate URL or downloads failed, try naive repair: check if file content is HTML with redirect to real pdf
        try:
            txt = p.read_text(errors="ignore")
            if "application/pdf" in txt or ".pdf" in txt:
                # try to find the first .pdf link inside file and re-download
                import re
                m = re.search(r"(https?://[^\s'\"<>]+\.pdf[^\s'\"<>]*)", txt)
                if m:
                    url = m.group(1)
                    print("  Found embedded pdf url in file, attempting download:", url)
                    out, status, info = download_and_save(url, p)
                    if out:
                        print("  Downloaded ->", out, "status:", status, "ctype:", info)
                        re_downloaded.append((p, url, out))
                        try:
                            p.unlink()
                        except Exception:
                            pass
                        continue
        except Exception:
            pass

        # as a last resort, keep file for manual inspection
        print("  Left as-is for manual inspection:", p)
        left.append(p)

    print("\nSummary:")
    print("Renamed files:", len(renamed))
    print("Re-downloaded files:", len(re_downloaded))
    print("Left for manual inspection:", len(left))
    if left:
        print("Manual files (first 10):")
        for a in left[:10]:
            print(" ", a)

if __name__ == "__main__":
    main()
