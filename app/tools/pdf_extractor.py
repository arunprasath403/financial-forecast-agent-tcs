"""
Safe PDF text extraction wrapper using pdfplumber with fallback to pdfminer text if needed.
If pdfplumber is not available or the PDF is scanned, extraction may be empty.
"""
from app.logger import logger

try:
    import pdfplumber
except Exception:
    pdfplumber = None

def extract_text_from_pdf(path: str) -> str:
    """
    Extracts and returns text content from the given PDF path.
    Returns empty string on failure.
    """
    if pdfplumber is None:
        logger.warning("pdfplumber not installed — attempting pdfminer fallback.")
        try:
            from pdfminer.high_level import extract_text
            txt = extract_text(path) or ""
            return txt
        except Exception as e:
            logger.error("pdfminer fallback failed: %s", e)
            return ""

    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    logger.debug("pdfplumber page extract error: %s", e)
                    page_text = ""
                text += page_text + "\n"
    except Exception as e:
        logger.error("pdfplumber open failed for %s: %s", path, e)
        # try pdfminer as fallback
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path) or ""
        except Exception as e2:
            logger.error("pdfminer fallback failed for %s: %s", path, e2)
            return ""
    return text
