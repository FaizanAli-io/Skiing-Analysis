from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
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
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    person = relationship("Person", back_populates="videos")
