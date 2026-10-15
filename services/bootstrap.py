"""Database set-up that runs once per deploy instead of on every start.

``python migrate.py`` applies everything. On start-up the API only compares
a schema fingerprint stored in the database with the code's fingerprint (one
quick query); when they differ it migrates automatically unless
RUN_MIGRATIONS_ON_STARTUP=false, in which case it logs how to fix it.
"""

import hashlib
import logging
import os
from datetime import datetime

from sqlalchemy import text

import models  # noqa: F401  (registers every table on Base.metadata)
from database import SCHEMA_STATEMENTS, Base, SessionLocal, engine, ensure_database_schema

logger = logging.getLogger(__name__)

# Arbitrary constant: lets concurrently starting processes take turns.
MIGRATION_LOCK_ID = 727_001


def schema_fingerprint() -> str:
    """Changes whenever a model column/index or a schema statement changes."""
    parts = list(SCHEMA_STATEMENTS)
    for table in sorted(Base.metadata.tables.values(), key=lambda t: t.name):
        parts.append(table.name)
        parts.extend(sorted(f"{column.name}:{column.type}" for column in table.columns))
        parts.extend(sorted(index.name or "" for index in table.indexes))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def _ensure_version_table(connection) -> None:
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS app_schema_version ("
        " id INTEGER PRIMARY KEY,"
        " version VARCHAR(64) NOT NULL,"
        " applied_at TIMESTAMP NOT NULL)"
    ))


def stored_fingerprint():
    try:
        with engine.connect() as connection:
            return connection.execute(text("SELECT version FROM app_schema_version WHERE id = 1")).scalar()
    except Exception:
        return None


def ensure_default_admin() -> None:
    from models.person import Person
    from services.auth import hash_password

    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return

    db = SessionLocal()
    try:
        email = admin_email.strip().lower()
        existing = db.query(Person).filter(Person.email == email).first()
        if existing:
            if existing.role != "admin":
                existing.role = "admin"
                db.commit()
            return
        db.add(Person(
            name=os.getenv("DEFAULT_ADMIN_NAME", "Bluerun Admin"),
            email=email,
            password_hash=hash_password(admin_password),
            role="admin",
            is_active=True,
        ))
        db.commit()
    finally:
        db.close()


def run_migrations() -> str:
    """Create tables, apply schema statements, seed the admin, record the version."""
    is_postgres = engine.dialect.name == "postgresql"
    with engine.connect() as lock_connection:
        if is_postgres:
            lock_connection.execute(text("SELECT pg_advisory_lock(:id)"), {"id": MIGRATION_LOCK_ID})
        try:
            Base.metadata.create_all(bind=engine)
            if is_postgres:
                ensure_database_schema()
            ensure_default_admin()
            fingerprint = schema_fingerprint()
            with engine.begin() as connection:
                _ensure_version_table(connection)
                connection.execute(text("DELETE FROM app_schema_version WHERE id = 1"))
                connection.execute(
                    text("INSERT INTO app_schema_version (id, version, applied_at) VALUES (1, :v, :t)"),
                    {"v": fingerprint, "t": datetime.utcnow()},
                )
            logger.info("Database schema is up to date (%s)", fingerprint)
            return fingerprint
        finally:
            if is_postgres:
                lock_connection.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": MIGRATION_LOCK_ID})
                lock_connection.commit()


def check_schema_on_startup() -> None:
    """One quick query on normal starts; migrate only when the schema changed."""
    if stored_fingerprint() == schema_fingerprint():
        return
    auto = os.getenv("RUN_MIGRATIONS_ON_STARTUP", "true").strip().lower() not in {"0", "false", "no", "off"}
    if auto:
        logger.warning("Database schema changed; applying migrations on start-up")
        run_migrations()
    else:
        logger.error(
            "Database schema is out of date. Run `python migrate.py` "
            "(RUN_MIGRATIONS_ON_STARTUP is false)."
        )
