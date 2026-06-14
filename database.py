from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool


load_dotenv()
SQLALCHEMY_DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("NEON_DATABASE_URL")
    or os.getenv("POSTGRE_SQL")
)

if not SQLALCHEMY_DATABASE_URL:
    raise RuntimeError("Database URL is missing. Set DATABASE_URL, NEON_DATABASE_URL, or POSTGRE_SQL in .env")



# Enhanced engine configuration with connection pooling and SSL handling
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,                    # Number of connections to keep open
    max_overflow=10,                # Additional connections when pool is full
    pool_pre_ping=True,             # Validate connections before use
    pool_recycle=3600,              # Recycle connections every hour (3600 seconds)
    pool_timeout=30,                # Timeout when getting connection from pool
    connect_args={
        "sslmode": "require",
        "connect_timeout": 30,
        "application_name": "ski-analysis-platform"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_database_schema():
    """Add new app columns to existing tables without dropping user data."""
    statements = [
        "ALTER TABLE persons ADD COLUMN IF NOT EXISTS phone_number VARCHAR(20)",
        "ALTER TABLE persons ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE persons ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS attempt_number INTEGER",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS display_mode VARCHAR(50)",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS overlay_renderer VARCHAR(50)",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS input_video_path VARCHAR(500)",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS output_video_path VARCHAR(500)",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS report_path VARCHAR(500)",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS blue_iq_score FLOAT",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS turns INTEGER",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS duration FLOAT",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS status VARCHAR(50)",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE video_analysis ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
