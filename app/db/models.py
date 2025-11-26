# app/db/models.py
from sqlalchemy import Column, Integer, Text, DateTime, String
from sqlalchemy.dialects.mysql import LONGTEXT
from datetime import datetime
from app.db.database import Base


class ForecastLog(Base):
    __tablename__ = "forecast_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # raw request payload (JSON string)
    request_payload = Column(LONGTEXT, nullable=True)

    # final response JSON (string)
    response_payload = Column(LONGTEXT, nullable=True)

    # optional source file path (e.g., /mnt/data/Task_ Financial Forecasting Agent for TCS.pdf)
    source_file_path = Column(String(1024), nullable=True, index=True)

    # small status string e.g. "success" or "error"
    status = Column(String(64), nullable=True)

    # optional error text if any
    error_text = Column(LONGTEXT, nullable=True)

    def as_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "request_payload": self.request_payload,
            "response_payload": self.response_payload,
            "source_file_path": self.source_file_path,
            "status": self.status,
            "error_text": self.error_text,
        }
