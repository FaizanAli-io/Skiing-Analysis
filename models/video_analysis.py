from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class VideoAnalysis(Base):
    __tablename__ = 'video_analysis'

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    video_name = Column(String(255))
    video_link = Column(String(500), nullable=True)

    pressure_score = Column(Float)
    balance_score = Column(Float)
    rotation_score = Column(Float)
    edging_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    person = relationship("Person", back_populates="videos")
