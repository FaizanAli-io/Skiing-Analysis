"""The notification bell: finished runs plus live progress of runs in processing."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.analysis_job import AnalysisJob
from models.notification import Notification
from models.person import Person
from models.video_analysis import VideoAnalysis
from services.auth import get_current_user
from services.notifications import version_label

router = APIRouter(prefix="/notifications", tags=["Notifications"])

# Processing jobs older than this are assumed abandoned and not shown.
ACTIVE_JOB_WINDOW = timedelta(hours=6)
MAX_ITEMS = 30


def _own(db: Session, notification_id: int, user: Person) -> Notification:
    item = db.query(Notification).filter(Notification.id == notification_id).first()
    if not item or item.person_id != user.id:
        raise HTTPException(status_code=404, detail="Notification not found")
    return item


@router.get("")
def list_notifications(current_user: Person = Depends(get_current_user), db: Session = Depends(get_db)):
    items = (
        db.query(Notification)
        .filter(Notification.person_id == current_user.id, Notification.dismissed_at.is_(None))
        .order_by(Notification.created_at.desc())
        .limit(MAX_ITEMS)
        .all()
    )
    jobs_query = db.query(AnalysisJob).filter(
        AnalysisJob.status.in_(("pending", "processing")),
        AnalysisJob.created_at >= datetime.utcnow() - ACTIVE_JOB_WINDOW,
    )
    if current_user.role != "admin":
        jobs_query = jobs_query.filter(AnalysisJob.person_id == current_user.id)
    jobs = jobs_query.order_by(AnalysisJob.created_at.asc()).all()

    # Download versions this user asked for that are still being made.
    preparing = []
    runs = db.query(VideoAnalysis).filter(
        VideoAnalysis.exports_status.in_(("requested", "processing")),
        VideoAnalysis.export_requesters.isnot(None),
    ).all()
    for run in runs:
        pending = set(run.pending_downloads)
        for key, people in (run.export_requesters or {}).items():
            if current_user.id in (people or []) and key in pending:
                mode, _, fmt = key.partition(":")
                preparing.append({
                    "analysis_id": run.id,
                    "mode": mode,
                    "format": fmt,
                    "label": version_label(mode, fmt),
                    "progress": int((run.export_progress or {}).get(key, 0)),
                })

    return {
        "unread": sum(1 for item in items if item.read_at is None),
        "preparing_videos": preparing,
        "active_jobs": [
            {
                "job_id": job.id,
                "status": job.status,
                "progress": int(job.progress or 0),
                "file_name": job.file_name if current_user.role == "admin" else None,
                "athlete_name": job.person.name if current_user.role == "admin" and job.person else None,
                "created_at": job.created_at,
            }
            for job in jobs
        ],
        "items": [
            {
                "id": item.id,
                "kind": item.kind,
                "emoji": item.emoji,
                "title": item.title,
                "body": item.body,
                "analysis_id": item.analysis_id,
                "athlete_id": item.athlete_id,
                "video_mode": item.video_mode,
                "video_format": item.video_format,
                "created_at": item.created_at,
                "read": item.read_at is not None,
                "downloaded": item.downloaded_at is not None,
            }
            for item in items
        ],
    }


@router.post("/read-all")
def mark_all_read(current_user: Person = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Notification).filter(
        Notification.person_id == current_user.id,
        Notification.read_at.is_(None),
    ).update({"read_at": datetime.utcnow()}, synchronize_session=False)
    db.commit()
    return {"ok": True}


@router.post("/{notification_id}/downloaded")
def mark_downloaded(notification_id: int, current_user: Person = Depends(get_current_user), db: Session = Depends(get_db)):
    item = _own(db, notification_id, current_user)
    now = datetime.utcnow()
    item.downloaded_at = item.downloaded_at or now
    item.read_at = item.read_at or now
    db.commit()
    return {"ok": True}


@router.post("/{notification_id}/dismiss")
def dismiss(notification_id: int, current_user: Person = Depends(get_current_user), db: Session = Depends(get_db)):
    item = _own(db, notification_id, current_user)
    item.dismissed_at = item.dismissed_at or datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.post("/clear")
def clear_all(current_user: Person = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Notification).filter(
        Notification.person_id == current_user.id,
        Notification.dismissed_at.is_(None),
    ).update({"dismissed_at": datetime.utcnow()}, synchronize_session=False)
    db.commit()
    return {"ok": True}
