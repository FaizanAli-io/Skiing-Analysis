from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from schemas.video_analysis import ClientAttemptOut
from schemas.analysis_timeline import AnalysisTimelineOut
from schemas.analysis_timeline import AnalysisTimelineOut
from schemas.personal_best import PersonalBestsResponse
from services.auth import get_current_user
from services.aws_s3 import S3Manager
from services.personal_bests import (
    build_performance_trends,
    completed_history_query,
    enrich_attempts,
    personal_bests_for_person,
)


router = APIRouter(prefix="/me", tags=["Client Portal"])


@router.get("/attempts", response_model=List[ClientAttemptOut])
def my_attempts(
    skip: int = 0,
    limit: int = 50,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempts = (
        db.query(VideoAnalysis)
        .filter(
            VideoAnalysis.person_id == current_user.id,
            or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)),
        )
        .order_by(VideoAnalysis.attempt_number.desc().nullslast(), VideoAnalysis.id.desc())
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

    enrich_attempts(db, attempts, client_numbering=True)
    return attempts


@router.get("/attempts/{attempt_id}", response_model=ClientAttemptOut)
def my_attempt_detail(
    attempt_id: int,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempt = (
        db.query(VideoAnalysis)
        .filter(
            VideoAnalysis.id == attempt_id,
            VideoAnalysis.person_id == current_user.id,
            or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)),
        )
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if S3Manager.is_enabled():
        if attempt.s3_video_key:
            attempt.video_link = S3Manager.get_video_url(attempt.s3_video_key, expiration=86400)
        if attempt.s3_report_key:
            attempt.report_path = S3Manager.get_report_url(attempt.s3_report_key, expiration=86400)
    enrich_attempts(db, [attempt], client_numbering=True)
    return attempt


@router.get("/attempts/{attempt_id}/timeline", response_model=AnalysisTimelineOut)
def my_attempt_timeline(
    attempt_id: int,
    resolution: Literal["second", "frame"] = "second",
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run graph data for one of the athlete's own (non-archived) runs."""
    from routes.admin_portal import timeline_response

    attempt = (
        db.query(VideoAnalysis)
        .filter(
            VideoAnalysis.id == attempt_id,
            VideoAnalysis.person_id == current_user.id,
            or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)),
        )
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return timeline_response(db, attempt, resolution)


@router.get("/attempts/{attempt_id}/timeline", response_model=AnalysisTimelineOut)
def my_attempt_timeline(
    attempt_id: int,
    resolution: Literal["second", "frame"] = "second",
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run graph data for one of the athlete's own (non-archived) runs."""
    from routes.admin_portal import timeline_response

    attempt = (
        db.query(VideoAnalysis)
        .filter(
            VideoAnalysis.id == attempt_id,
            VideoAnalysis.person_id == current_user.id,
            or_(VideoAnalysis.is_archived.is_(False), VideoAnalysis.is_archived.is_(None)),
        )
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return timeline_response(db, attempt, resolution)


@router.get("/personal-bests", response_model=PersonalBestsResponse)
def my_personal_bests(
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {
        "person_id": current_user.id,
        "personal_bests": personal_bests_for_person(db, current_user.id, client_numbering=True),
    }


@router.get("/trends")
def get_performance_trends(
    limit: Optional[int] = None,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run history, with the latest run compared against the first run."""
    return build_performance_trends(completed_history_query(db, current_user.id).all(), limit, client_numbering=True)
