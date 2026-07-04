from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String(36), primary_key=True, index=True)  # UUID
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    video_analysis_id = Column(Integer, ForeignKey("video_analysis.id"), nullable=True)
    
    # Job metadata
    file_name = Column(String(255), nullable=False)
    display_mode = Column(String(50), default="coach")
    generate_report = Column(Integer, default=1)  # 1 = True, 0 = False
    
    # Status tracking
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    person = relationship("Person", back_populates="analysis_jobs")
    video_analysis = relationship("VideoAnalysis", foreign_keys=[video_analysis_id])
    
    def __repr__(self):
        return f"<AnalysisJob(id={self.id}, status={self.status}, person_id={self.person_id})>"
