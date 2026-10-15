from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        Index("ix_analysis_jobs_person_id", "person_id"),
        Index("ix_analysis_jobs_status_created", "status", "created_at"),
        Index("ix_analysis_jobs_runner_status", "runner", "status", "created_at"),
    )

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

    # Queue: which runner owns the job ("inline" API process or "worker"),
    # what kind it is (upload, drive, s3) and the parameters to run it with.
    runner = Column(String(20), nullable=True)
    job_kind = Column(String(20), nullable=True)
    payload = Column(JSON, nullable=True)
    attempts = Column(Integer, default=0)
    # Machine that created the job; a worker only runs its own machine's jobs.
    runner_host = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    person = relationship("Person", back_populates="analysis_jobs")
    video_analysis = relationship("VideoAnalysis", foreign_keys=[video_analysis_id])
    
    def __repr__(self):
        return f"<AnalysisJob(id={self.id}, status={self.status}, person_id={self.person_id})>"
