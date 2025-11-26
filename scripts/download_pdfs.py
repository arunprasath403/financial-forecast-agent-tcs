# scripts/download_all_pdfs.py  (REPLACEMENT FOR download_last_3_quarters.py)

import os
import requests
import hashlib
from urllib.parse import urlparse, unquote
from pathlib import Path

LINKS_FILE = Path("data/docs/screener_links.txt")
OUT_DIR = Path("data/docs/screener_pdfs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TIMEOUT = 30


def safe_name(url: str) -> str:
    # build safe filename using short hash + base name
    parsed = urlparse(url)
    base = os.path.basename(unquote(parsed.path)) or "doc.pdf"
    base = base.split("?")[0]
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{h}_{base}"


def load_links():
    if not LINKS_FILE.exists():
        print("Links file not found:", LINKS_FILE)
        return []
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def download_all(links):
    saved = []
    for url in links:
        name = safe_name(url)
        outpath = OUT_DIR / name

        # Skip if already downloaded
        if outpath.exists():
            print("exists:", outpath)
            saved.append(str(outpath))
            continue

        try:
            print("downloading:", url)
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            r.raise_for_status()

            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print("saved:", outpath)
            saved.append(str(outpath))

        except Exception as e:
            print("failed to download:", url, "->", e)

    return saved


def main():
    links = load_links()
    if not links:
        print("No links found.")
        return

    print(f"Found {len(links)} links — downloading all...")
    files = download_all(links)

    print("\nDownloaded:", len(files), "files")
    print("Output folder:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
