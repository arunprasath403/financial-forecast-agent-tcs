#!/usr/bin/env python3
"""
scripts/fix_aspx_files.py
Make .aspx-saved files into .pdf safely (collision-resistant).
"""
import logging
from pathlib import Path

log = logging.getLogger("fix_aspx_files")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# ---- repo-root resolver ----
def find_repo_root(start: Path = None, markers=("data", ".git")) -> Path:
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(10):
        for m in markers:
            if (cur / m).exists():
                return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(__file__).resolve().parents[1]

REPO_ROOT = find_repo_root()
PDF_DIR = REPO_ROOT / "data" / "docs" / "screener_pdfs"

# ---- helpers ----
def unique_target(path: Path) -> Path:
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

def try_rename_pdf_file(p: Path) -> Path:
    """
    Rename p -> p_stem_pdf.pdf (or p_stem_pdf(n).pdf if collision).
    Returns the final Path used (or original Path on error).
    """
    try:
        candidate = p.with_name(f"{p.stem}_pdf.pdf")
        if candidate.exists():
            new_target = unique_target(candidate)
            log.warning("Target %s already exists — using unique name %s", candidate, new_target)
            candidate = new_target

        p.rename(candidate)
        log.info("Renamed %s -> %s", p, candidate)
        return candidate

    except FileNotFoundError:
        log.warning("Source file not found (skipping): %s", p)
        return p
    except PermissionError as e:
        log.exception("Permission denied renaming %s -> %s : %s", p, candidate, e)
        return p
    except Exception as e:
        log.exception("Unexpected error renaming %s : %s", p, e)
        return p

# ---- main ----
def main():
    if not PDF_DIR.exists():
        log.error("PDF_DIR not found: %s", PDF_DIR)
        return

    for p in sorted(PDF_DIR.iterdir()):
        if p.is_dir():
            continue
        # rename .aspx/AnnPdfOpen files to *_pdf.pdf safely
        if p.suffix and p.suffix.lower() != ".pdf":
            # some files already have _pdf.pdf variants - try safe rename
            newp = try_rename_pdf_file(p)
            if newp and newp != p:
                log.info("Fixed: %s -> %s", p, newp)

if __name__ == "__main__":
    main()
