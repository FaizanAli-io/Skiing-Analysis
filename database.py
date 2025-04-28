from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Update with your MySQL credentials
SQLALCHEMY_DATABASE_URL = (
    "postgresql+psycopg2://neondb_owner:npg_rOInV0TUa9HF@ep-sweet-bush-a4q9wya6-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()