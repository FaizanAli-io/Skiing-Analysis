import os
import logging
from typing import Optional
from database import SessionLocal
from services.new_analysis import analyze_video
from services.aws_s3 import S3Manager, upload_analysis_files
import crud.person as person_crud
import crud.job as job_crud
from models.video_analysis import VideoAnalysis
from datetime import date

logger = logging.getLogger(__name__)


def update_job_progress(job_id: str, progress: int):
    """Update job progress in database"""
    db = SessionLocal()
    try:
        job_crud.update_job_status(db, job_id, status="processing", progress=progress)
    except Exception as e:
        logger.error(f"Failed to update job progress for {job_id}: {e}")
    finally:
        db.close()


def process_video_analysis_background(
    job_id: str,
    file_path: str,
    user_id: int,
    display_mode: str = "coach",
    report: bool = False,
    user_name: Optional[str] = None,
    attempt_number: Optional[int] = None,
):
    """
    Background task to process video analysis.
    Updates job status throughout the process.
    """
    db = SessionLocal()
    
    try:
        logger.info(f"Starting background analysis for job {job_id}")
        
        # Update status to processing
        job_crud.update_job_status(db, job_id, status="processing", progress=5)
        
        # Get user info if not provided
        if not user_name or attempt_number is None:
            db_user = person_crud.get_person(db, user_id)
            if not db_user:
                raise Exception(f"User {user_id} not found")
            user_name = db_user.name
            if attempt_number is None:
                from crud.video_analysis import get_next_attempt_number
                attempt_number = get_next_attempt_number(db, user_id)
        
        session_date = date.today().isoformat()
        
        # Progress: 10% - Starting analysis
        update_job_progress(job_id, 10)
        
        # Run video analysis
        logger.info(f"Job {job_id}: Running video analysis")
        results = analyze_video(
            file_path,
            display_mode=display_mode,
            overlay_renderer="premium",
            report=report,
            user_name=user_name,
            attempt_number=attempt_number,
            session_date=session_date,
        )
        
        # Progress: 60% - Analysis complete
        update_job_progress(job_id, 60)
        
        # Upload to S3 if enabled
        s3_video_url = None
        s3_report_url = None
        s3_snapshot_url = None
        s3_uploads = {}
        
        if S3Manager.is_enabled():
            logger.info(f"Job {job_id}: Uploading to S3")
            s3_uploads = upload_analysis_files(
                video_path=results.get("output_path"),
                report_path=results.get("report_path"),
                snapshot_path=results.get("snapshot_path")
            )
            s3_video_url = s3_uploads.get("video_url")
            s3_report_url = s3_uploads.get("report_url")
            s3_snapshot_url = s3_uploads.get("snapshot_url")
        
        # Progress: 80% - S3 upload complete
        update_job_progress(job_id, 80)
        
        # Use S3 URLs if available
        if s3_video_url:
            results["video_url"] = s3_video_url
        if s3_report_url:
            results["report_url"] = s3_report_url
        
        # Fallback to local URLs if S3 not enabled
        if not s3_video_url and "output_path" in results:
            results["video_url"] = f"/outputs/{os.path.basename(results['output_path'])}"
        if not s3_report_url and "report_path" in results:
            results["report_url"] = f"/outputs/{os.path.basename(results['report_path'])}"
        
        # Convert numpy floats to Python floats for database compatibility
        blue_iq_score = float((
            results["pressure_score"]
            + results["balance_score"]
            + results["rotation_score"]
            + results["edging_score"]
        ) / 4)
        
        pressure_score = float(results["pressure_score"])
        balance_score = float(results["balance_score"])
        rotation_score = float(results["rotation_score"])
        edging_score = float(results["edging_score"])
        turns = int(results.get("turns", 0))
        duration = float(results.get("duration", 0.0))
        
        # Save to database
        logger.info(f"Job {job_id}: Saving to database")
        attempt = VideoAnalysis(
            person_id=user_id,
            attempt_number=attempt_number,
            video_name=os.path.basename(file_path),
            video_link=results.get("video_url"),
            input_video_path=file_path,
            output_video_path=results.get("output_path"),
            report_path=s3_report_url if s3_report_url else results.get("report_path"),
            s3_video_key=s3_uploads.get("video_s3_key") if S3Manager.is_enabled() else None,
            s3_report_key=s3_uploads.get("report_s3_key") if S3Manager.is_enabled() else None,
            s3_snapshot_key=s3_uploads.get("snapshot_s3_key") if S3Manager.is_enabled() else None,
            display_mode=display_mode,
            overlay_renderer="premium",
            blue_iq_score=blue_iq_score,
            pressure_score=pressure_score,
            balance_score=balance_score,
            rotation_score=rotation_score,
            edging_score=edging_score,
            turns=turns,
            duration=duration,
            status="completed",
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        
        # Progress: 100% - Complete
        job_crud.update_job_status(
            db,
            job_id,
            status="completed",
            progress=100,
            video_analysis_id=attempt.id
        )
        
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        job_crud.update_job_status(
            db,
            job_id,
            status="failed",
            error_message=str(e)
        )
    finally:
        db.close()
