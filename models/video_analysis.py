from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class VideoAnalysis(Base):
    __tablename__ = 'video_analysis'

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    attempt_number = Column(Integer, nullable=True)
    video_name = Column(String(255))
    video_link = Column(String(500), nullable=True)
    input_video_path = Column(String(500), nullable=True)
    output_video_path = Column(String(500), nullable=True)
    report_path = Column(String(500), nullable=True)
    s3_video_key = Column(String(500), nullable=True)  # S3 key for video
    s3_report_key = Column(String(500), nullable=True)  # S3 key for report
    s3_snapshot_key = Column(String(500), nullable=True)  # S3 key for snapshot
    # Extra downloadable versions rendered in the background, keyed
    # {mode: {format: {"s3_key": ...} | {"path": ...}}}; see services/video_exports.py.
    video_exports = Column(JSON, nullable=True)
    exports_status = Column(String(20), nullable=True)  # processing, ready, failed
    exports_updated_at = Column(DateTime, nullable=True)
    display_mode = Column(String(50), nullable=True)
    overlay_renderer = Column(String(50), nullable=True)

    blue_iq_score = Column(Float)
    pressure_score = Column(Float)
    balance_score = Column(Float)
    rotation_score = Column(Float)
    edging_score = Column(Float)
    turns = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    status = Column(String(50), default="completed")
    instructor_notes = Column(Text, nullable=True)
    notes_source = Column(String(20), nullable=True)  # ai, template, or coach
    is_archived = Column(Boolean, default=False, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    person = relationship("Person", back_populates="videos")
    timeline = relationship(
        "AnalysisTimeline",
        back_populates="analysis",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @property
    def main_video_mode(self) -> str:
        return self.display_mode if self.display_mode in ("athlete", "coach") else "athlete"

    @property
    def available_downloads(self) -> dict:
        """{mode: [formats]} that can be downloaded right now."""
        available = {"athlete": [], "coach": []}
        if self.s3_video_key or self.output_video_path or self.video_link:
            available[self.main_video_mode].append("original")
        for mode, formats in (self.video_exports or {}).items():
            for fmt in formats or {}:
                if mode in available and fmt not in available[mode]:
                    available[mode].append(fmt)
        order = ("original", "vertical", "square")
        return {mode: sorted(formats, key=order.index) for mode, formats in available.items()}

    @property
    def downloads_pending(self) -> bool:
        from services.video_exports import is_pending

        return is_pending(self.exports_status, self.exports_updated_at)

    @property
    def person_name(self):
        if hasattr(self, "_person_name") and self._person_name is not None:
            return self._person_name
        return self.person.name if self.person else None

    @person_name.setter
    def person_name(self, value):
        self._person_name = value

    @property
    def client_name(self):
        if hasattr(self, "_client_name") and self._client_name is not None:
            return self._client_name
        return self.person.name if self.person else None

    @client_name.setter
    def client_name(self, value):
        self._client_name = value

