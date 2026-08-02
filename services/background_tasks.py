import logging
import os
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from sqlalchemy.exc import OperationalError

from database import SessionLocal
from services.new_analysis import analyze_video
from services.aws_s3 import S3Manager, upload_analysis_files
import crud.person as person_crud
import crud.job as job_crud
from models.video_analysis import VideoAnalysis
from models.analysis_timeline import AnalysisTimeline
from services.personal_bests import pre_run_personal_best_context


logger = logging.getLogger(__name__)

DB_MAX_ATTEMPTS = 3
DB_RETRY_DELAY_SECONDS = 0.5
T = TypeVar("T")


def _rollback_quietly(db: Any) -> None:
    try:
        db.rollback()
    except Exception:
        # A disconnected DBAPI connection may also reject rollback. Closing the
        # session below invalidates it; the next attempt receives a fresh one.
        pass


def _close_quietly(db: Any) -> None:
    try:
        db.close()
    except Exception:
        # Closing a transaction whose socket already disappeared can itself
        # raise; it must not replace the original error or prevent a retry.
        pass


def _run_db_operation(
    operation: Callable[[Any], T],
    operation_name: str,
    max_attempts: int = DB_MAX_ATTEMPTS,
) -> T:
    """Run a short DB operation in a fresh session with disconnect retries."""
    for attempt_index in range(1, max_attempts + 1):
        db = SessionLocal()
        try:
            return operation(db)
        except OperationalError:
            _rollback_quietly(db)
            if attempt_index >= max_attempts:
                raise
            logger.warning(
                "%s lost its database connection; retrying with a fresh "
                "session (%s/%s)",
                operation_name,
                attempt_index + 1,
                max_attempts,
            )
            time.sleep(DB_RETRY_DELAY_SECONDS * attempt_index)
        except Exception:
            _rollback_quietly(db)
            raise
        finally:
            _close_quietly(db)

    raise RuntimeError(f"{operation_name} exhausted its database retries")


def update_job_progress(job_id: str, progress: int) -> bool:
    """Update progress using a short, reconnect-safe database session."""

    def operation(db: Any) -> None:
        job = job_crud.update_job_status(
            db,
            job_id,
            status="processing",
            progress=progress,
        )
        if job is None:
            raise RuntimeError(f"Analysis job {job_id} was not found")

    try:
        _run_db_operation(operation, f"Update progress for job {job_id}")
        return True
    except Exception as exc:
        logger.error("Failed to update job progress for %s: %s", job_id, exc)
        return False


def _prepare_job_context(
    job_id: str,
    user_id: int,
    user_name: Optional[str],
    attempt_number: Optional[int],
    session_date: str,
) -> Tuple[str, int, Dict[str, Any]]:
    """Resolve athlete metadata, then release the DB before video processing."""

    def operation(db: Any) -> Tuple[str, int, Dict[str, Any]]:
        job = job_crud.update_job_status(
            db,
            job_id,
            status="processing",
            progress=5,
        )
        if job is None:
            raise RuntimeError(f"Analysis job {job_id} was not found")

        resolved_name = user_name
        resolved_attempt = attempt_number
        if not resolved_name or resolved_attempt is None:
            db_user = person_crud.get_person(db, user_id)
            if not db_user:
                raise RuntimeError(f"User {user_id} not found")
            resolved_name = resolved_name or db_user.name
            if resolved_attempt is None:
                from crud.video_analysis import get_next_attempt_number

                resolved_attempt = get_next_attempt_number(db, user_id)

        personal_best_context = pre_run_personal_best_context(
            db,
            user_id,
            session_date,
        )
        return str(resolved_name), int(resolved_attempt), personal_best_context

    return _run_db_operation(operation, f"Prepare job {job_id}")


def _build_analysis_record(
    file_path: str,
    user_id: int,
    attempt_number: int,
    display_mode: str,
    results: Dict[str, Any],
    s3_uploads: Dict[str, Any],
    s3_enabled: bool,
    s3_report_url: Optional[str],
) -> Dict[str, Any]:
    pressure_score = float(results["pressure_score"])
    balance_score = float(results["balance_score"])
    rotation_score = float(results["rotation_score"])
    edging_score = float(results["edging_score"])

    return {
        "person_id": user_id,
        "attempt_number": attempt_number,
        "video_name": os.path.basename(file_path),
        "video_link": results.get("video_url"),
        "input_video_path": file_path,
        "output_video_path": results.get("output_path"),
        "report_path": (
            s3_report_url if s3_report_url else results.get("report_path")
        ),
        "s3_video_key": s3_uploads.get("video_s3_key") if s3_enabled else None,
        "s3_report_key": s3_uploads.get("report_s3_key") if s3_enabled else None,
        "s3_snapshot_key": (
            s3_uploads.get("snapshot_s3_key") if s3_enabled else None
        ),
        "display_mode": display_mode,
        "overlay_renderer": "premium",
        "blue_iq_score": float(
            (
                pressure_score
                + balance_score
                + rotation_score
                + edging_score
            )
            / 4
        ),
        "pressure_score": pressure_score,
        "balance_score": balance_score,
        "rotation_score": rotation_score,
        "edging_score": edging_score,
        "turns": int(results.get("turns", 0)),
        "duration": float(results.get("duration", 0.0)),
        "status": "completed",
    }


def _save_completed_analysis(
    job_id: str,
    analysis_values: Dict[str, Any],
    timeline_values: Optional[Dict[str, Any]] = None,
) -> int:
    """Atomically save the analysis and mark its job completed."""

    def operation(db: Any) -> int:
        job = job_crud.get_job(db, job_id)
        if job is None:
            raise RuntimeError(f"Analysis job {job_id} was not found")

        # Makes a retry safe if the server committed just before the client
        # observed a disconnect.
        if job.video_analysis_id is not None:
            existing = (
                db.query(VideoAnalysis)
                .filter(VideoAnalysis.id == job.video_analysis_id)
                .first()
            )
            if existing is not None:
                if timeline_values:
                    existing_timeline = (
                        db.query(AnalysisTimeline)
                        .filter(
                            AnalysisTimeline.video_analysis_id == existing.id
                        )
                        .first()
                    )
                    if existing_timeline is None:
                        db.add(AnalysisTimeline(
                            video_analysis_id=existing.id,
                            **timeline_values,
                        ))
                job.status = "completed"
                job.progress = 100
                job.error_message = None
                job.completed_at = job.completed_at or datetime.utcnow()
                db.commit()
                return int(existing.id)

        attempt = VideoAnalysis(**analysis_values)
        db.add(attempt)
        db.flush()

        if timeline_values:
            db.add(AnalysisTimeline(
                video_analysis_id=attempt.id,
                **timeline_values,
            ))

        job.video_analysis_id = attempt.id
        job.status = "completed"
        job.progress = 100
        job.error_message = None
        job.completed_at = datetime.utcnow()

        # The analysis row and job link either both commit or both roll back.
        db.commit()
        return int(attempt.id)

    return _run_db_operation(operation, f"Save completed job {job_id}")


def _mark_job_failed(job_id: str, error: Exception) -> None:
    """Record failure through a clean session, never the failed transaction."""
    error_message = str(error)[:4000]

    def operation(db: Any) -> None:
        job = job_crud.update_job_status(
            db,
            job_id,
            status="failed",
            error_message=error_message,
        )
        if job is None:
            raise RuntimeError(f"Analysis job {job_id} was not found")

    try:
        _run_db_operation(operation, f"Mark job {job_id} failed")
    except Exception as status_error:
        logger.error(
            "Could not mark job %s as failed after fresh-session retries: %s",
            job_id,
            status_error,
            exc_info=True,
        )


def process_video_analysis_background(
    job_id: str,
    file_path: str,
    user_id: int,
    display_mode: str = "coach",
    report: bool = False,
    user_name: Optional[str] = None,
    attempt_number: Optional[int] = None,
):
    """Process one video without holding a DB session during long work."""
    try:
        logger.info("Starting background analysis for job %s", job_id)
        session_date = date.today().isoformat()
        user_name, attempt_number, personal_best_context = _prepare_job_context(
            job_id,
            user_id,
            user_name,
            attempt_number,
            session_date,
        )

        update_job_progress(job_id, 10)

        logger.info("Job %s: Running video analysis", job_id)
        results = analyze_video(
            file_path,
            display_mode=display_mode,
            overlay_renderer="premium",
            report=report,
            user_name=user_name,
            attempt_number=attempt_number,
            session_date=session_date,
            session_number=personal_best_context["session_number"],
            previous_personal_bests=personal_best_context["personal_bests"],
        )

        update_job_progress(job_id, 60)

        s3_video_url = None
        s3_report_url = None
        s3_uploads: Dict[str, Any] = {}
        s3_enabled = S3Manager.is_enabled()
        if s3_enabled:
            logger.info("Job %s: Uploading to S3", job_id)
            s3_uploads = upload_analysis_files(
                video_path=results.get("output_path"),
                report_path=results.get("report_path"),
                snapshot_path=results.get("snapshot_path"),
            )
            s3_video_url = s3_uploads.get("video_url")
            s3_report_url = s3_uploads.get("report_url")

        update_job_progress(job_id, 80)

        if s3_video_url:
            results["video_url"] = s3_video_url
        elif "output_path" in results:
            results["video_url"] = (
                f"/outputs/{os.path.basename(results['output_path'])}"
            )

        if s3_report_url:
            results["report_url"] = s3_report_url
        elif "report_path" in results:
            results["report_url"] = (
                f"/outputs/{os.path.basename(results['report_path'])}"
            )

        analysis_values = _build_analysis_record(
            file_path=file_path,
            user_id=user_id,
            attempt_number=attempt_number,
            display_mode=display_mode,
            results=results,
            s3_uploads=s3_uploads,
            s3_enabled=s3_enabled,
            s3_report_url=s3_report_url,
        )

        logger.info("Job %s: Saving to database", job_id)
        timeline_values = results.get("analysis_timeline")
        analysis_id = _save_completed_analysis(
            job_id,
            analysis_values,
            timeline_values=timeline_values,
        )
        logger.info(
            "Job %s: Completed successfully as analysis %s",
            job_id,
            analysis_id,
        )

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        _mark_job_failed(job_id, exc)
