from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text

from database import Base


class Notification(Base):
    """A message in someone's notification bell (e.g. "your new run is ready")."""

    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notifications_person_open", "person_id", "dismissed_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(30), nullable=False)  # run_ready, run_failed
    emoji = Column(String(16), nullable=True)
    title = Column(String(200), nullable=False)
    body = Column(Text, nullable=True)
    analysis_id = Column(Integer, ForeignKey("video_analysis.id", ondelete="CASCADE"), nullable=True)
    athlete_id = Column(Integer, nullable=True)  # for admin links to the athlete profile
    # Which download version a "video_ready" notification is about.
    video_mode = Column(String(20), nullable=True)
    video_format = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    read_at = Column(DateTime, nullable=True)
    downloaded_at = Column(DateTime, nullable=True)
    dismissed_at = Column(DateTime, nullable=True)
