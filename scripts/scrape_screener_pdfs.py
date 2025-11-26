# scripts/scrape_screener_pdfs.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os, sys, traceback

SCREENER_URL = "https://www.screener.in/company/TCS/consolidated/#documents"
BASE_URL = "https://www.screener.in"
OUTPUT = "data/docs/screener_links.txt"
os.makedirs("data/docs", exist_ok=True)

def scrape_screener_pdfs(url):
    print("Fetching:", url)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=20)
        print("HTTP status:", r.status_code)
        if r.status_code != 200:
            print("Non-200 response, aborting.")
            return []
        text = r.text
        # debug: print first 800 chars so we can inspect if JS-rendered or not
        print("HTML preview (first 800 chars):")
        print(text[:800].replace("\n"," "))
        soup = BeautifulSoup(text, "html.parser")

        # First attempt: documents div
        docs_div = soup.find("div", {"id": "documents"})
        if docs_div:
            print("Found #documents div")
            anchors = docs_div.find_all("a", href=True)
        else:
            print("No #documents div — falling back to scanning all <a> tags")
            anchors = soup.find_all("a", href=True)

        links = []
        for a in anchors:
            href = a["href"]
            if not href:
                continue
            # make absolute
            full = urljoin(BASE_URL, href)
            if ".pdf" in full.lower():
                links.append(full)
        links = list(dict.fromkeys(links))
        return links
    except Exception as e:
        print("Exception during scrape:", e)
        traceback.print_exc()
        return []

if __name__ == "__main__":
    found = scrape_screener_pdfs(SCREENER_URL)
    print("\nTotal PDF links found:", len(found))
    if found:
        print("First 10 links:")
        for i,l in enumerate(found[:10]):
            print(i+1, l)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in found:
            f.write(p + "\n")
    print("Saved to", OUTPUT)
