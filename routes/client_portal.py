from datetime import date, datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from schemas.video_analysis import VideoAnalysisOut
from schemas.personal_best import PersonalBestsResponse
from services.auth import get_current_user
from services.aws_s3 import S3Manager
from services.personal_bests import (
    METRIC_KEYS,
    build_personal_best_history,
    display_score,
    enrich_attempts,
    personal_bests_for_person,
)


router = APIRouter(prefix="/me", tags=["Client Portal"])


def _attempt_date_label(attempt: VideoAnalysis) -> str:
    value = attempt.created_at or attempt.timestamp
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10] if value else "Unknown date"


@router.get("/attempts", response_model=List[VideoAnalysisOut])
def my_attempts(
    skip: int = 0,
    limit: int = 50,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempts = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.person_id == current_user.id)
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

    enrich_attempts(db, attempts)
    return attempts


@router.get("/attempts/{attempt_id}", response_model=VideoAnalysisOut)
def my_attempt_detail(
    attempt_id: int,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempt = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.id == attempt_id, VideoAnalysis.person_id == current_user.id)
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    enrich_attempts(db, [attempt])
    return attempt


@router.get("/personal-bests", response_model=PersonalBestsResponse)
def my_personal_bests(
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {
        "person_id": current_user.id,
        "personal_bests": personal_bests_for_person(db, current_user.id),
    }


@router.get("/trends")
def get_performance_trends(
    limit: int = 15,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return neutral run history with current-to-personal-best context."""

    all_attempts = (
        db.query(VideoAnalysis)
        .filter(
            VideoAnalysis.person_id == current_user.id,
            (VideoAnalysis.status == "completed") | (VideoAnalysis.status.is_(None)),
        )
        .order_by(VideoAnalysis.created_at.asc().nullsfirst(), VideoAnalysis.id.asc())
        .all()
    )

    if not all_attempts:
        return {"error": "No completed runs are available yet", "total_runs": 0}

    history = build_personal_best_history(all_attempts)
    latest = all_attempts[-1]
    latest_comparisons = history["comparisons_by_attempt"].get(latest.id, {})
    chart_limit = max(1, min(int(limit), 100))
    attempts = all_attempts[-chart_limit:]
    dates = [_attempt_date_label(attempt) for attempt in attempts]
    runs = [int(attempt.attempt_number or attempt.id) for attempt in attempts]
    time_series = {
        "dates": dates,
        "runs": runs,
        "turns": [attempt.turns or 0 for attempt in attempts],
    }
    for metric in METRIC_KEYS:
        time_series[metric] = [
            display_score(
                attempt.blue_iq_score
                if metric == "blue_iq"
                else getattr(attempt, f"{metric}_score")
            ) or 0
            for attempt in attempts
        ]

    return {
        "total_runs": len(all_attempts),
        "displayed_runs": len(attempts),
        "date_range": {
            "start": dates[0],
            "end": dates[-1]
        },
        "time_series": time_series,
        "personal_bests": history["personal_bests"],
        "current_vs_personal_best": latest_comparisons,
    }
