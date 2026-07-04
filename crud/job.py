from sqlalchemy.orm import Session
from models.analysis_job import AnalysisJob
from datetime import datetime
from typing import Optional, List


def create_job(
    db: Session,
    job_id: str,
    person_id: int,
    file_name: str,
    display_mode: str = "coach",
    generate_report: bool = True
) -> AnalysisJob:
    """Create a new analysis job"""
    job = AnalysisJob(
        id=job_id,
        person_id=person_id,
        file_name=file_name,
        display_mode=display_mode,
        generate_report=1 if generate_report else 0,
        status="pending",
        progress=0
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[AnalysisJob]:
    """Get a job by ID"""
    return db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()


def update_job_status(
    db: Session,
    job_id: str,
    status: str,
    progress: Optional[int] = None,
    error_message: Optional[str] = None,
    video_analysis_id: Optional[int] = None
) -> Optional[AnalysisJob]:
    """Update job status and progress"""
    job = get_job(db, job_id)
    if not job:
        return None
    
    job.status = status
    
    if progress is not None:
        job.progress = progress
    
    if error_message is not None:
        job.error_message = error_message
    
    if video_analysis_id is not None:
        job.video_analysis_id = video_analysis_id
    
    # Update timestamps
    if status == "processing" and not job.started_at:
        job.started_at = datetime.utcnow()
    
    if status in ["completed", "failed"]:
        job.completed_at = datetime.utcnow()
        job.progress = 100 if status == "completed" else job.progress
    
    db.commit()
    db.refresh(job)
    return job


def get_all_jobs(
    db: Session,
    person_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[AnalysisJob]:
    """Get all jobs with optional filters"""
    query = db.query(AnalysisJob)
    
    if person_id is not None:
        query = query.filter(AnalysisJob.person_id == person_id)
    
    if status is not None:
        query = query.filter(AnalysisJob.status == status)
    
    return query.order_by(AnalysisJob.created_at.desc()).offset(skip).limit(limit).all()


def delete_old_jobs(db: Session, days: int = 7) -> int:
    """Delete completed/failed jobs older than specified days"""
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    deleted = db.query(AnalysisJob).filter(
        AnalysisJob.status.in_(["completed", "failed"]),
        AnalysisJob.completed_at < cutoff_date
    ).delete()
    
    db.commit()
    return deleted
