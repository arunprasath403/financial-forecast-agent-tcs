# app/db/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Example: mysql+pymysql://user:password@host:3306/dbname
DATABASE_URL = os.getenv("FORECAST_DB_URL")  # set this in env, recommended

if not DATABASE_URL:
    # fallback local default for quick dev (change DB name & creds as needed)
    DATABASE_URL = os.getenv(
        "MYSQL_URL",
        "mysql+pymysql://root:password@127.0.0.1:3306/forecast_agent"
    )

# create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def init_db():
    """
    Create DB tables (calls Base.metadata.create_all).
    Call this at startup or manually once.
    """
    from app.db import models  # ensure models are imported
    Base.metadata.create_all(bind=engine)
