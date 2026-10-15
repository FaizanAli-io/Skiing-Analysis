"""Where analysis jobs run: inside the API process or in the separate worker.

JOB_RUNNER=inline (default) keeps the original behaviour: FastAPI runs the job
in the API process right after the response. JOB_RUNNER=worker only records
the job; ``worker.py`` (its own service) claims and runs jobs one at a time,
so the website stays responsive while videos process.

Every job stores its kind and parameters, plus the runner it was created for.
A worker only claims jobs created for the worker *on the same machine*
(the API and worker run side by side), so a worker started on a developer
machine can never pick up production jobs, even when it shares the database.
"""

import logging
import os
import re
import socket
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

TEMP_VIDEO_DIR = os.path.abspath("temp_videos")

RUNNER_INLINE = "inline"
RUNNER_WORKER = "worker"
MAX_ATTEMPTS = 2  # first run + one retry after an interrupted worker


def this_host() -> str:
    return os.getenv("WORKER_HOST_NAME") or socket.gethostname()


def runner_mode() -> str:
    mode = os.getenv("JOB_RUNNER", RUNNER_INLINE).strip().lower()
    return RUNNER_WORKER if mode == RUNNER_WORKER else RUNNER_INLINE


def _runners() -> Dict[str, Callable[..., None]]:
    from services import background_tasks

    return {
        "upload": background_tasks.process_video_analysis_background,
        "drive": background_tasks.process_google_drive_analysis_background,
        "s3": background_tasks.process_s3_upload_background,
    }


def run_job(job_id: str, kind: str, payload: Dict[str, Any]) -> None:
    _runners()[kind](job_id=job_id, **payload)


def dispatch(background_tasks, job_id: str, kind: str, payload: Dict[str, Any]) -> None:
    """Run now in this process, or leave it queued for the worker."""
    if kind not in _runners():
        raise ValueError(f"Unknown job kind: {kind}")
    if runner_mode() == RUNNER_INLINE:
        background_tasks.add_task(run_job, job_id, kind, payload)


def scrub_payload(job_id: str, *keys: str) -> None:
    """Remove secrets (e.g. a Google OAuth token) once they are no longer needed."""
    from database import SessionLocal, close_session_quietly
    from models.analysis_job import AnalysisJob

    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job and job.payload:
            job.payload = {k: v for k, v in job.payload.items() if k not in keys}
            db.commit()
    except Exception as exc:
        logger.warning("Could not scrub job %s payload: %s", job_id, exc)
    finally:
        close_session_quietly(db)


def claim_next_job(db) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Atomically take the oldest pending worker job, or return None."""
    from models.analysis_job import AnalysisJob

    candidates = (
        db.query(AnalysisJob.id)
        .filter(
            AnalysisJob.status == "pending",
            AnalysisJob.runner == RUNNER_WORKER,
            AnalysisJob.runner_host == this_host(),
        )
        .order_by(AnalysisJob.created_at.asc())
        .limit(5)
        .all()
    )
    for (job_id,) in candidates:
        # Only one process can flip pending -> processing for a given row.
        claimed = db.execute(
            text(
                "UPDATE analysis_jobs SET status = 'processing', started_at = :now, "
                "attempts = COALESCE(attempts, 0) + 1 "
                "WHERE id = :id AND status = 'pending'"
            ),
            {"id": job_id, "now": datetime.utcnow()},
        ).rowcount
        db.commit()
        if claimed == 1:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            return job.id, job.job_kind, dict(job.payload or {})
    return None


def recover_interrupted_jobs(db) -> int:
    """Requeue worker jobs left 'processing' by a crash/restart, at most once."""
    from models.analysis_job import AnalysisJob

    stuck = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.status == "processing",
            AnalysisJob.runner == RUNNER_WORKER,
            AnalysisJob.runner_host == this_host(),
        )
        .all()
    )
    for job in stuck:
        if (job.attempts or 0) < MAX_ATTEMPTS:
            job.status = "pending"
            job.progress = 0
            logger.warning("Requeued interrupted job %s", job.id)
        else:
            job.status = "failed"
            job.error_message = "Processing was interrupted. Please upload the video again."
            job.completed_at = datetime.utcnow()
            logger.error("Job %s failed after repeated interruptions", job.id)
    db.commit()
    return len(stuck)


def queue_job(
    background_tasks,
    job_id: str,
    kind: str,
    user_id: int,
    file_name: str,
    display_mode: str,
    report: bool,
    payload: dict,
) -> None:
    """Record the job with everything needed to run it, then run or queue it."""
    import crud.job as job_crud
    from database import SessionLocal, close_session_quietly

    write_db = SessionLocal()
    try:
        job_crud.create_job(
            write_db,
            job_id=job_id,
            person_id=user_id,
            file_name=file_name,
            display_mode=display_mode,
            generate_report=report,
            runner=runner_mode(),
            job_kind=kind,
            payload=payload,
            runner_host=this_host(),
        )
    finally:
        close_session_quietly(write_db)
    dispatch(background_tasks, job_id, kind, payload)


def safe_upload_name(name: Optional[str], fallback: str) -> str:
    clean = re.sub(r"[^\w\-_\.]", "_", os.path.basename(str(name or "")).strip()) or fallback
    if not clean.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v")):
        clean += ".mp4"
    return clean
