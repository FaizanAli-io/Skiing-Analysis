from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from schemas.person import PersonOut
from schemas.video_analysis import VideoAnalysisOut
from services.auth import require_admin
from services.aws_s3 import S3Manager


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
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    attempts = (
        db.query(VideoAnalysis)
        .order_by(VideoAnalysis.created_at.desc().nullslast(), VideoAnalysis.id.desc())
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
    
    return attempts


@router.get("/users/{user_id}/attempts", response_model=List[VideoAnalysisOut])
def admin_user_attempts(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(Person).filter(Person.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    attempts = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.person_id == user_id)
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
    
    return attempts

