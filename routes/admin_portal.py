import os
import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from models.analysis_timeline import AnalysisTimeline
from models.analysis_job import AnalysisJob
from schemas.analysis_timeline import AnalysisTimelineOut
from schemas.person import PersonOut
from schemas.video_analysis import VideoAnalysisOut
from schemas.personal_best import LeaderboardsResponse, PersonalBestsResponse
from services.auth import require_admin
from services.aws_s3 import S3Manager, AWS_S3_BUCKET
from services.personal_bests import (
    build_leaderboards,
    enrich_attempts,
    personal_bests_for_person,
)
from services.analysis_timeline import (
    aggregate_timeline_samples,
    sanitize_parameter_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Portal"])


@router.get("/users", response_model=List[PersonOut])
def admin_list_users(
    skip: int = 0,
    limit: int = 100,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return db.query(Person).order_by(Person.created_at.desc().nullslast(), Person.id.desc()).offset(skip).limit(limit).all()


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


@router.get("/attempts", response_model=List[VideoAnalysisOut])
def admin_list_attempts(
    skip: int = 0,
    limit: int = 100,
    include_archived: bool = Query(False, description="Include archived runs"),
    only_archived: bool = Query(False, description="Only return archived runs"),
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(VideoAnalysis)
    if only_archived:
        query = query.filter(VideoAnalysis.is_archived.is_(True))
    elif not include_archived:
        query = query.filter(or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)))

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
        for key in (attempt.s3_video_key, attempt.s3_report_key, attempt.s3_snapshot_key):
            if key:
                try:
                    S3Manager.delete_file(AWS_S3_BUCKET, key)
                except Exception as exc:
                    logger.warning(f"Failed to delete S3 file {key}: {exc}")

    # 4. Clean up local files
    for path in (attempt.input_video_path, attempt.output_video_path, attempt.report_path):
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
