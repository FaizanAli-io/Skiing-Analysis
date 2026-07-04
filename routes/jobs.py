from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models.person import Person
from services.auth import get_current_user, require_admin
from schemas.job import JobStatus, JobStatusWithResult
import crud.job as job_crud
from typing import List, Optional

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=JobStatusWithResult)
def get_job_status(
    job_id: str,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the status of an analysis job.
    Users can only see their own jobs, admins can see all jobs.
    """
    job = job_crud.get_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check permissions
    if current_user.role != "admin" and job.person_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this job"
        )
    
    response = JobStatusWithResult(
        job_id=job.id,
        person_id=job.person_id,
        file_name=job.file_name,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=None
    )
    
    # Include video analysis result if completed
    if job.status == "completed" and job.video_analysis:
        response.result = {
            "id": job.video_analysis.id,
            "attempt_number": job.video_analysis.attempt_number,
            "video_link": job.video_analysis.video_link,
            "report_path": job.video_analysis.report_path,
            "blue_iq_score": job.video_analysis.blue_iq_score,
            "pressure_score": job.video_analysis.pressure_score,
            "balance_score": job.video_analysis.balance_score,
            "rotation_score": job.video_analysis.rotation_score,
            "edging_score": job.video_analysis.edging_score,
            "turns": job.video_analysis.turns,
            "duration": job.video_analysis.duration,
        }
    
    return response


@router.get("/", response_model=List[JobStatus])
def list_jobs(
    status: Optional[str] = None,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all jobs for the current user.
    Optionally filter by status.
    """
    person_id = None if current_user.role == "admin" else current_user.id
    jobs = job_crud.get_all_jobs(db, person_id=person_id, status=status)
    
    return [
        JobStatus(
            job_id=job.id,
            person_id=job.person_id,
            file_name=job.file_name,
            status=job.status,
            progress=job.progress,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@router.delete("/{job_id}")
def delete_job(
    job_id: str,
    current_user: Person = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Delete a job (admin only).
    Only completed or failed jobs can be deleted.
    """
    job = job_crud.get_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.status not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only completed or failed jobs can be deleted"
        )
    
    db.delete(job)
    db.commit()
    
    return {"message": "Job deleted successfully"}
