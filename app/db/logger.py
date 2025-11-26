# app/db/logger.py
import json
import traceback
from contextlib import contextmanager
from typing import Optional

from app.db.database import SessionLocal, init_db
from app.db.models import ForecastLog
from app.logger import logger  # your existing logger

# ensure DB tables exist
def ensure_db():
    try:
        init_db()
    except Exception as e:
        logger.exception("init_db failed: %s", e)


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def log_request_and_response(
    request_payload: Optional[dict],
    response_payload: Optional[dict],
    source_file_path: Optional[str] = "/data/Task_ Financial Forecasting Agent for TCS.pdf",
    status: str = "success",
    error_text: Optional[str] = None,
) -> int:
    """
    Writes a ForecastLog row. Returns inserted row id.
    By default, sets the source_file_path to the uploaded assessment PDF path.
    """

    ensure_db()

    req_str = None
    res_str = None
    try:
        req_str = json.dumps(request_payload, ensure_ascii=False) if request_payload is not None else None
        res_str = json.dumps(response_payload, ensure_ascii=False) if response_payload is not None else None
    except Exception as e:
        # fallback to repr
        req_str = repr(request_payload)
        res_str = repr(response_payload)

    log_id = None
    with get_db_session() as db:
        try:
            row = ForecastLog(
                request_payload=req_str,
                response_payload=res_str,
                source_file_path=source_file_path,
                status=status,
                error_text=error_text,
            )
            db.add(row)
            db.flush()  # populates id
            log_id = row.id
            logger.info("Saved forecast log id=%s status=%s", log_id, status)
        except Exception as e:
            logger.exception("Failed to write forecast log: %s", e)
            raise

    return log_id
