import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db
import crud.video_analysis as crud
import schemas.video_analysis as schemas
from models.person import Person
from models.video_analysis import VideoAnalysis
from services.auth import (
    create_download_token,
    decode_download_token,
    get_current_user,
    require_admin,
)
from services.aws_s3 import S3Manager
from services.personal_bests import client_run_number
from services.video_exports import FORMATS, MODES


router = APIRouter(prefix="/videos", tags=["Video Analysis"])


@router.get("/{video_id}", response_model=schemas.VideoAnalysisOut)
def read(video_id: int, _admin: Person = Depends(require_admin), db: Session = Depends(get_db)):
    db_video = crud.get_video(db, video_id)
    if db_video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return db_video


def _video_sources(video: VideoAnalysis, mode: str, fmt: str, run_number: Optional[int] = None):
    """Return (s3_key, local_path, filename) for one downloadable version."""
    run_number = run_number or video.attempt_number or video.id
    filename = f"bluerun_run_{run_number}_{mode}_{fmt}.mp4"
    if fmt == "original" and mode == video.main_video_mode:
        return video.s3_video_key, video.output_video_path or video.input_video_path, filename
    entry = ((video.video_exports or {}).get(mode) or {}).get(fmt) or {}
    return entry.get("s3_key"), entry.get("path"), filename


def _authorized_video(db: Session, video_id: int, user: Person) -> VideoAnalysis:
    video = crud.get_video(db, video_id)
    # Clients may only reach their own, non-archived runs; 404 avoids
    # revealing which IDs exist.
    if not video or (
        user.role != "admin"
        and (video.person_id != user.id or video.is_archived)
    ):
        raise HTTPException(status_code=404, detail="Video not found")
    return video


def _validate(value: Optional[str], allowed, name: str, default: str) -> str:
    value = (value or default).lower()
    if value not in allowed:
        raise HTTPException(status_code=422, detail=f"{name} must be one of: {', '.join(allowed)}")
    return value


@router.get("/{video_id}/download-url")
def get_download_url(
    video_id: int,
    format: str = Query("original"),
    mode: Optional[str] = Query(None, description="athlete or coach; defaults to the run's own mode"),
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a short-lived link the browser can open to download the video."""
    video = _authorized_video(db, video_id, current_user)
    fmt = _validate(format, FORMATS, "format", "original")
    mode = _validate(mode, MODES, "mode", video.main_video_mode)
    run_number = client_run_number(db, video) if current_user.role != "admin" else None
    s3_key, local_path, filename = _video_sources(video, mode, fmt, run_number)

    if S3Manager.is_enabled() and s3_key:
        download_url = S3Manager.get_video_download_url(s3_key, filename=filename, expiration=600)
        if download_url:
            return {"download_url": download_url, "filename": filename}

    if local_path and os.path.exists(local_path):
        token = create_download_token(current_user.id, video.id, fmt, mode, run_number)
        return {
            "download_url": f"/api/videos/{video.id}/download?format={fmt}&mode={mode}&token={token}",
            "filename": filename,
        }

    if video.downloads_pending:
        raise HTTPException(status_code=409, detail="This version is still being prepared. Please try again in a minute.")
    raise HTTPException(status_code=404, detail="This version is not available for this run")


@router.get("/{video_id}/download")
def download_video(
    video_id: int,
    token: str = Query(...),
    format: str = Query("original"),
    mode: str = Query(...),
    db: Session = Depends(get_db),
):
    """Serve a local video file using a link issued by /download-url."""
    fmt = _validate(format, FORMATS, "format", "original")
    mode = _validate(mode, MODES, "mode", "athlete")
    claims = decode_download_token(token)
    if claims.get("vid") != video_id or claims.get("fmt") != fmt or claims.get("mode") != mode:
        raise HTTPException(status_code=403, detail="Download link does not match this video")

    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    s3_key, local_path, filename = _video_sources(video, mode, fmt, claims.get("run"))

    if local_path and os.path.exists(local_path):
        return FileResponse(local_path, media_type="video/mp4", filename=filename)
    if S3Manager.is_enabled() and s3_key:
        download_url = S3Manager.get_video_download_url(s3_key, filename=filename, expiration=600)
        if download_url:
            return RedirectResponse(url=download_url, status_code=307)
    raise HTTPException(status_code=404, detail="Video file not found")


@router.get("/", response_model=List[schemas.VideoAnalysisOut])
def read_all(skip: int = 0, limit: int = 10, _admin: Person = Depends(require_admin), db: Session = Depends(get_db)):
    return crud.get_all_videos(db, skip, limit)


@router.delete("/{video_id}")
def delete(video_id: int, _admin: Person = Depends(require_admin), db: Session = Depends(get_db)):
    deleted = crud.delete_video(db, video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"message": "Deleted successfully"}
