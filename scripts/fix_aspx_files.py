from pathlib import Path
import logging

log = logging.getLogger("fix_aspx_files")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

# repo-root resolver (keep your existing function or this one)
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

def unique_target_for_pdf(base_target: Path) -> Path:
    """
    Given a desired target Path (usually ending with .pdf), return a Path that does not exist.
    If base_target exists, try base_target -> base_target(1) -> base_target(2) ... before returning.
    """
    if not base_target.exists():
        return base_target
    parent = base_target.parent
    stem = base_target.stem
    suffix = base_target.suffix or ".pdf"
    i = 1
    while True:
        candidate = parent / f"{stem}({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def try_rename_pdf_file(p: Path) -> Path:
    """
    Attempt to rename `p` to a *_pdf.pdf name. If that name exists, pick a unique one
    using unique_target_for_pdf(). Returns the final Path used (or p on error/no-op).
    This function will NOT raise on existing-target collisions.
    """
    try:
        if not p.exists():
            log.warning("Source not found, skipping: %s", p)
            return p

        # produce desired new name: original_stem + "_pdf.pdf"
        desired = p.with_name(f"{p.stem}_pdf.pdf")

        # if desired exists (or for safety always), pick a non-existing unique path
        final_target = unique_target_for_pdf(desired)

        # if final_target is same as p (rare), avoid self-rename
        if final_target.resolve() == p.resolve():
            log.debug("Final target same as source for %s; skipping rename", p)
            return p

        # rename (Windows will raise if target exists; we've ensured it does not)
        p.rename(final_target)
        log.info("Renamed: %s -> %s", p, final_target)
        return final_target

    except PermissionError as e:
        log.exception("Permission denied renaming %s -> %s: %s", p, desired, e)
        return p
    except FileNotFoundError:
        log.warning("File disappeared during rename attempt (skipping): %s", p)
        return p
    except Exception as e:
        # catch-all: log and continue (do not raise)
        log.exception("Unexpected error renaming %s: %s", p, e)
        return p

# --------------------------
# Example main renaming loop (replace your existing loop)
# --------------------------
def main():
    if not PDF_DIR.exists():
        log.error("PDF_DIR not found: %s", PDF_DIR)
        return

    for p in sorted(PDF_DIR.iterdir()):
        if p.is_dir():
            continue

        # rename only non-.pdf files or files with weird suffixes
        if p.suffix and p.suffix.lower() != ".pdf":
            newp = try_rename_pdf_file(p)
            if newp and newp != p:
                log.info("Fixed: %s -> %s", p, newp)
        else:
            # ensure .pdf files really are .pdf (optional check)
            log.debug("Already PDF (or no rename needed): %s", p)

if __name__ == "__main__":
    main()
