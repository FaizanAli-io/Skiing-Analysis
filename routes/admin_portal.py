import os
import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from models.analysis_timeline import AnalysisTimeline
from models.analysis_job import AnalysisJob
from schemas.analysis_timeline import AnalysisTimelineOut
from schemas.person import AdminPersonOut, PersonOut
from schemas.video_analysis import VideoAnalysisOut
from schemas.personal_best import LeaderboardsResponse, PersonalBestsResponse
from services.auth import require_admin
from services.coach_notes import MAX_NOTE_CHARS
from services.video_exports import export_entries
from services.aws_s3 import S3Manager, AWS_S3_BUCKET
from services.personal_bests import (
    build_leaderboards,
    build_performance_trends,
    completed_history_query,
    enrich_attempts,
    personal_bests_for_person,
    report_context_for_attempt,
)
from services.analysis_timeline import (
    aggregate_timeline_samples,
    sanitize_parameter_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Portal"])


@router.get("/users", response_model=List[AdminPersonOut])
def admin_list_users(
    skip: int = 0,
    limit: int = 100,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    users = db.query(Person).order_by(Person.created_at.desc().nullslast(), Person.id.desc()).offset(skip).limit(limit).all()
    # Count every non-archived run, independent of what the dashboard has loaded.
    counts = dict(
        db.query(VideoAnalysis.person_id, func.count(VideoAnalysis.id))
        .filter(
            VideoAnalysis.person_id.in_([user.id for user in users]),
            or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)),
        )
        .group_by(VideoAnalysis.person_id)
        .all()
    ) if users else {}
    return [
        AdminPersonOut.model_validate(user).model_copy(update={"run_count": counts.get(user.id, 0)})
        for user in users
    ]


@router.get("/users/{user_id}", response_model=PersonOut)
def admin_get_user(
    user_id: int,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(Person).filter(Person.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def _naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """created_at is stored as naive UTC; compare like with like."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


@router.get("/attempts", response_model=List[VideoAnalysisOut])
def admin_list_attempts(
    skip: int = 0,
    limit: int = 100,
    include_archived: bool = Query(False, description="Include archived runs"),
    only_archived: bool = Query(False, description="Only return archived runs"),
    start: Optional[datetime] = Query(None, description="Only runs created at or after this time (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="Only runs created before this time (ISO 8601)"),
    q: Optional[str] = Query(None, max_length=100, description="Athlete name or email contains"),
    person_ids: Optional[str] = Query(None, max_length=2000, description="Comma-separated athlete IDs (fuzzy search results)"),
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(VideoAnalysis)
    if only_archived:
        query = query.filter(VideoAnalysis.is_archived.is_(True))
    elif not include_archived:
        query = query.filter(or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)))
    if start is not None:
        query = query.filter(VideoAnalysis.created_at >= _naive_utc(start))
    if end is not None:
        query = query.filter(VideoAnalysis.created_at < _naive_utc(end))
    if person_ids is not None:
        try:
            ids = [int(part) for part in person_ids.split(",") if part.strip()]
        except ValueError:
            raise HTTPException(status_code=422, detail="person_ids must be comma-separated integers")
        query = query.filter(VideoAnalysis.person_id.in_(ids))
    term = (q or "").strip()
    if term:
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        query = query.join(Person, Person.id == VideoAnalysis.person_id).filter(
            or_(
                Person.name.ilike(pattern, escape="\\"),
                Person.email.ilike(pattern, escape="\\"),
            )
        )

    attempts = (
        query.order_by(VideoAnalysis.created_at.desc().nullslast(), VideoAnalysis.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    # Refresh S3 presigned URLs if S3 is enabled
    if S3Manager.is_enabled():
        for attempt in attempts:
            if attempt.s3_video_key:
                attempt.video_link = S3Manager.get_video_url(attempt.s3_video_key, expiration=86400)
            if attempt.s3_report_key:
                attempt.report_path = S3Manager.get_report_url(attempt.s3_report_key, expiration=86400)

    enrich_attempts(db, attempts)
    return attempts


@router.get("/users/{user_id}/attempts", response_model=List[VideoAnalysisOut])
def admin_user_attempts(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    include_archived: bool = Query(False, description="Include archived runs"),
    only_archived: bool = Query(False, description="Only return archived runs"),
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(Person).filter(Person.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    query = db.query(VideoAnalysis).filter(VideoAnalysis.person_id == user_id)
    if only_archived:
        query = query.filter(VideoAnalysis.is_archived.is_(True))
    elif not include_archived:
        query = query.filter(or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)))

    attempts = (
        query.order_by(VideoAnalysis.attempt_number.desc().nullslast(), VideoAnalysis.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    # Refresh S3 presigned URLs if S3 is enabled
    if S3Manager.is_enabled():
        for attempt in attempts:
            if attempt.s3_video_key:
                attempt.video_link = S3Manager.get_video_url(attempt.s3_video_key, expiration=86400)
            if attempt.s3_report_key:
                attempt.report_path = S3Manager.get_report_url(attempt.s3_report_key, expiration=86400)

    enrich_attempts(db, attempts)
    return attempts


@router.post("/attempts/{attempt_id}/archive")
def admin_archive_attempt(
    attempt_id: int,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    attempt = db.query(VideoAnalysis).filter(VideoAnalysis.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    
    attempt.is_archived = True
    db.commit()
    return {"message": "Attempt archived successfully", "id": attempt_id, "is_archived": True}


@router.post("/attempts/{attempt_id}/restore")
def admin_restore_attempt(
    attempt_id: int,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    attempt = db.query(VideoAnalysis).filter(VideoAnalysis.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    
    attempt.is_archived = False
    db.commit()
    return {"message": "Attempt restored successfully", "id": attempt_id, "is_archived": False}


@router.delete("/attempts/{attempt_id}")
def admin_delete_attempt(
    attempt_id: int,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    attempt = db.query(VideoAnalysis).filter(VideoAnalysis.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    # 1. Unlink any analysis jobs referencing this video analysis
    db.query(AnalysisJob).filter(AnalysisJob.video_analysis_id == attempt_id).update({"video_analysis_id": None})

    # 2. Delete associated timeline if exists
    db.query(AnalysisTimeline).filter(AnalysisTimeline.video_analysis_id == attempt_id).delete()

    # 3. Clean up S3 files if enabled
    if S3Manager.is_enabled() and AWS_S3_BUCKET:
        for key in (
            attempt.s3_video_key,
            attempt.s3_report_key,
            attempt.s3_snapshot_key,
            *export_entries(attempt, "s3_key"),
        ):
            if key:
                try:
                    S3Manager.delete_file(AWS_S3_BUCKET, key)
                except Exception as exc:
                    logger.warning(f"Failed to delete S3 file {key}: {exc}")

    # 4. Clean up local files
    for path in (
        attempt.input_video_path,
        attempt.output_video_path,
        attempt.report_path,
        *export_entries(attempt, "path"),
    ):
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"Deleted local file: {path}")
            except Exception as exc:
                logger.warning(f"Failed to delete local file {path}: {exc}")

    # 5. Delete the VideoAnalysis row
    db.delete(attempt)
    db.commit()

    return {"message": "Attempt permanently deleted", "id": attempt_id}


@router.get(
    "/users/{user_id}/personal-bests",
    response_model=PersonalBestsResponse,
)
def admin_user_personal_bests(
    user_id: int,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(Person).filter(Person.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "person_id": user_id,
        "personal_bests": personal_bests_for_person(db, user_id),
    }


@router.get("/users/{user_id}/trends")
def admin_user_trends(
    user_id: int,
    limit: Optional[int] = None,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """The same run history the athlete sees in their own portal."""
    if not db.query(Person).filter(Person.id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found")
    return build_performance_trends(completed_history_query(db, user_id).all(), limit)


@router.get("/leaderboards", response_model=LeaderboardsResponse)
def admin_leaderboards(
    limit: int = 50,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return one personal-best entry per athlete for every score metric."""
    return {"leaderboards": build_leaderboards(db, limit=limit)}


@router.get(
    "/attempts/{analysis_id}/timeline",
    response_model=AnalysisTimelineOut,
)
def admin_attempt_timeline(
    analysis_id: int,
    resolution: Literal["second", "frame"] = "second",
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return structured coach metrics for a run, averaged by second by default."""
    analysis = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.id == analysis_id)
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    timeline = (
        db.query(AnalysisTimeline)
        .filter(AnalysisTimeline.video_analysis_id == analysis_id)
        .first()
    )
    if timeline is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Detailed timeline is unavailable for this run. "
                "Generate a new analysis to record graph data."
            ),
        )

    samples = aggregate_timeline_samples(
        timeline.samples or [],
        resolution=resolution,
    )
    duration = float(analysis.duration or 0.0)
    if duration <= 0 and samples:
        duration = float(samples[-1].get("time_seconds") or 0.0)

    return AnalysisTimelineOut(
        analysis_id=analysis.id,
        resolution=resolution,
        duration_seconds=duration,
        sample_rate_hz=float(timeline.sample_rate_hz),
        scoring_version=timeline.scoring_version,
        parameter_config=sanitize_parameter_config(timeline.parameter_config),
        samples=samples,
    )


class InstructorNotesPayload(BaseModel):
    # Matches what fits in the PDF note box.
    notes: Optional[str] = Field(default=None, max_length=MAX_NOTE_CHARS)


@router.patch("/attempts/{attempt_id}/notes", response_model=VideoAnalysisOut)
def update_instructor_notes(
    attempt_id: int,
    payload: InstructorNotesPayload,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Save the coach's edit of the run note and regenerate the PDF with it.

    Notes start as the automatic AI coach note; once a coach saves an edit the
    note is marked ``coach`` and is never overwritten by generation again.
    """
    attempt = db.query(VideoAnalysis).filter(VideoAnalysis.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    cleaned_notes = payload.notes.strip() if payload.notes else None
    attempt.instructor_notes = cleaned_notes
    attempt.notes_source = "coach" if cleaned_notes else None

    downloaded_snapshot = None
    try:
        from services.report_generator import generate_basic_report

        timeline_entry = db.query(AnalysisTimeline).filter(AnalysisTimeline.video_analysis_id == attempt_id).first()
        score_timeline = timeline_entry.samples if timeline_entry else []

        os.makedirs("outputs", exist_ok=True)
        video_filename = os.path.basename(attempt.output_video_path or attempt.video_name or f"attempt_{attempt_id}.mp4")
        base_name = os.path.splitext(video_filename)[0]
        local_output_path = os.path.join("outputs", f"{base_name}.mp4")

        # The pipeline removes local copies once they are in S3, so fetch the
        # snapshot back or the regenerated report would lose its photo.
        local_snapshot = None
        cand_snapshot = os.path.join("outputs", f"{base_name}_snapshot.jpg")
        if os.path.exists(cand_snapshot):
            local_snapshot = cand_snapshot
        elif attempt.s3_snapshot_key and S3Manager.is_enabled() and AWS_S3_BUCKET:
            if S3Manager.download_file(AWS_S3_BUCKET, attempt.s3_snapshot_key, cand_snapshot):
                local_snapshot = downloaded_snapshot = cand_snapshot

        result_payload = {
            "output_path": local_output_path,
            "pressure_score": attempt.pressure_score or 0.0,
            "balance_score": attempt.balance_score or 0.0,
            "rotation_score": attempt.rotation_score or 0.0,
            "edging_score": attempt.edging_score or 0.0,
            "duration": attempt.duration or 0.0,
            "turns": attempt.turns or 0,
            "snapshot_path": local_snapshot,
            "instructor_notes": cleaned_notes,
        }

        pb_context = report_context_for_attempt(db, attempt)
        report_context = {
            "user_name": attempt.person.name if attempt.person else "Client",
            # The PDF is client-facing: use the athlete's own run number.
            "attempt_number": pb_context["client_run_number"],
            "session_date": pb_context["session_date"],
            "session_number": pb_context["session_number"],
            "previous_personal_bests": pb_context["previous_personal_bests"],
            "first_run_blue_iq": pb_context["first_run_blue_iq"],
            "instructor_notes": cleaned_notes,
        }

        report_result = generate_basic_report(
            result_payload,
            score_timeline,
            context=report_context,
        )
        new_report_path = report_result.get("report_path")
        if new_report_path and os.path.exists(new_report_path):
            attempt.report_path = new_report_path
            if S3Manager.is_enabled():
                # Overwrite the existing object so old report keys are not orphaned.
                s3_key = S3Manager.upload_report(new_report_path, s3_key=attempt.s3_report_key)
                if s3_key:
                    attempt.s3_report_key = s3_key
                    attempt.report_path = S3Manager.get_report_url(s3_key, expiration=86400)
                    try:
                        os.remove(new_report_path)
                    except OSError:
                        pass
    except Exception as exc:
        logger.exception("Could not regenerate PDF report when saving instructor notes: %s", exc)
    finally:
        if downloaded_snapshot and os.path.exists(downloaded_snapshot):
            try:
                os.remove(downloaded_snapshot)
            except OSError:
                pass

    db.commit()
    db.refresh(attempt)
    if S3Manager.is_enabled() and attempt.s3_video_key:
        attempt.video_link = S3Manager.get_video_url(attempt.s3_video_key, expiration=86400)
    enrich_attempts(db, [attempt])
    return attempt
