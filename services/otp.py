"""Issue and verify one-time email codes (hashed, expiring, attempt-limited)."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.email_otp import EmailOTP
from services.auth import _jwt_secret

CODE_TTL_MINUTES = 10
RESEND_COOLDOWN_SECONDS = 60
MAX_ATTEMPTS = 5


def _hash_code(email: str, purpose: str, code: str) -> str:
    message = f"{purpose}:{email}:{code}".encode("utf-8")
    return hmac.new(_jwt_secret().encode("utf-8"), message, hashlib.sha256).hexdigest()


def _active_codes(db: Session, email: str, purpose: str):
    return db.query(EmailOTP).filter(
        EmailOTP.email == email,
        EmailOTP.purpose == purpose,
        EmailOTP.consumed_at.is_(None),
    )


def issue_code(db: Session, email: str, purpose: str, payload: Optional[Dict[str, Any]] = None) -> str:
    """Create a fresh 6-digit code, replacing any earlier unused one."""
    now = datetime.utcnow()
    latest = _active_codes(db, email, purpose).order_by(EmailOTP.created_at.desc()).first()
    if latest and (now - latest.created_at).total_seconds() < RESEND_COOLDOWN_SECONDS:
        wait = RESEND_COOLDOWN_SECONDS - int((now - latest.created_at).total_seconds())
        raise HTTPException(status_code=429, detail=f"Please wait {wait} seconds before requesting another code.")

    _active_codes(db, email, purpose).update({"consumed_at": now}, synchronize_session=False)
    code = f"{secrets.randbelow(1_000_000):06d}"
    db.add(EmailOTP(
        email=email,
        purpose=purpose,
        code_hash=_hash_code(email, purpose, code),
        payload=payload,
        attempts=0,
        expires_at=now + timedelta(minutes=CODE_TTL_MINUTES),
        created_at=now,
    ))
    db.commit()
    return code


def verify_code(db: Session, email: str, purpose: str, code: str) -> EmailOTP:
    """Consume a valid code or raise 400. Wrong guesses count toward a limit."""
    now = datetime.utcnow()
    record = _active_codes(db, email, purpose).order_by(EmailOTP.created_at.desc()).first()
    invalid = HTTPException(status_code=400, detail="That code is invalid or has expired. Please request a new one.")
    if not record or record.expires_at < now or record.attempts >= MAX_ATTEMPTS:
        raise invalid

    if not hmac.compare_digest(record.code_hash, _hash_code(email, purpose, str(code or "").strip())):
        record.attempts += 1
        if record.attempts >= MAX_ATTEMPTS:
            record.consumed_at = now
        db.commit()
        raise invalid

    record.consumed_at = now
    db.commit()
    return record
