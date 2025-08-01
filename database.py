from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool


load_dotenv()
# Update with your MySQL credentials
SQLALCHEMY_DATABASE_URL = os.getenv("POSTGRE_SQL")



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
        "application_name": "ai-booking-agent"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()