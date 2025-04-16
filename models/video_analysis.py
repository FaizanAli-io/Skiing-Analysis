from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

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

    person = relationship("Person", back_populates="videos")
