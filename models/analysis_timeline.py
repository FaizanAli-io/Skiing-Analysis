from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from database import Base


class AnalysisTimeline(Base):
    """Versioned, frame-level metric data for one completed analysis."""

    __tablename__ = "analysis_timelines"

    id = Column(Integer, primary_key=True, index=True)
    video_analysis_id = Column(
        Integer,
        ForeignKey("video_analysis.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    sample_rate_hz = Column(Float, nullable=False, default=10.0)
    scoring_version = Column(String(80), nullable=False)
    samples = Column(JSON, nullable=False)
    parameter_config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    analysis = relationship("VideoAnalysis", back_populates="timeline")
