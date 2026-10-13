from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String

from database import Base


class EmailOTP(Base):
    """One-time email codes for signup verification and password resets."""

    __tablename__ = "email_otps"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), nullable=False, index=True)
    purpose = Column(String(30), nullable=False, index=True)  # signup, password_reset
    code_hash = Column(String(128), nullable=False)
    # Pending signup details (name, phone, password hash) until the code is verified.
    payload = Column(JSON, nullable=True)
    attempts = Column(Integer, default=0, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    consumed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
