"""Direct-to-S3 uploads (DIRECT_S3_UPLOADS=true).

1. POST /uploads/presign  -> signed form for one new object under uploads/
2. the browser uploads the file straight to S3 (no load on this server)
3. POST /analyze-s3/      -> verifies the object and queues the analysis
"""

import os
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

import crud.person as person_crud
import crud.video_analysis as video_crud
from database import SessionLocal, close_session_quietly
from models.person import Person
from services.auth import require_admin
from services.aws_s3 import S3Manager
from services.job_queue import TEMP_VIDEO_DIR, queue_job, safe_upload_name

router = APIRouter(tags=["Uploads"])


UPLOAD_PREFIX = "uploads/"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _direct_uploads_enabled() -> bool:
    flag = os.getenv("DIRECT_S3_UPLOADS", "false").strip().lower() in {"1", "true", "yes", "on"}
    return flag and S3Manager.is_enabled()


def _max_upload_bytes() -> int:
    try:
        return int(float(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
    except ValueError:
        return 2048 * 1024 * 1024


@router.get("/config/uploads")
def get_upload_config():
    """Tells the upload form whether to send files straight to S3."""
    return {"direct_s3": _direct_uploads_enabled(), "max_upload_mb": _max_upload_bytes() // (1024 * 1024)}


@router.post("/uploads/presign")
async def presign_upload(request: Request, current_admin: Person = Depends(require_admin)):
    """Step 1 of a direct upload: a signed form for one new object under uploads/."""
    if not _direct_uploads_enabled():
        raise HTTPException(status_code=400, detail="Direct uploads are not enabled")
    payload = await request.json()
    original_name = str(payload.get("file_name") or "")
    file_name = safe_upload_name(original_name, "upload.mp4")
    content_type = str(payload.get("content_type") or "").strip().lower()
    # Judge by what the browser sent, before the name is normalised.
    if not content_type.startswith("video/") and not original_name.lower().endswith(VIDEO_EXTENSIONS):
        raise HTTPException(status_code=422, detail="Please upload a video file")
    if not content_type.startswith("video/"):
        content_type = "video/mp4"
    try:
        size = int(payload.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0 or size > _max_upload_bytes():
        raise HTTPException(status_code=422, detail=f"Videos must be under {_max_upload_bytes() // (1024 * 1024)} MB")

    key = f"{UPLOAD_PREFIX}{uuid.uuid4().hex}/{file_name}"
    signed = S3Manager.presign_upload(key, content_type, _max_upload_bytes())
    if not signed:
        raise HTTPException(status_code=502, detail="Could not prepare the upload. Please try again.")
    return {"key": key, "url": signed["url"], "fields": signed["fields"]}


@router.post("/analyze-s3/")
async def analyze_uploaded_video(
    request: Request,
    background_tasks: BackgroundTasks,
    current_admin: Person = Depends(require_admin),
):
    """Step 2 of a direct upload: queue analysis of the object the browser uploaded."""
    payload = await request.json()
    s3_key = str(payload.get("s3_key") or "")
    if not s3_key.startswith(UPLOAD_PREFIX) or ".." in s3_key or s3_key.count("/") != 2:
        raise HTTPException(status_code=422, detail="Invalid upload reference")
    size = S3Manager.object_size(s3_key)
    if size is None:
        raise HTTPException(status_code=404, detail="The uploaded video was not found. Please upload it again.")
    if size > _max_upload_bytes():
        raise HTTPException(status_code=422, detail="The uploaded video is too large")

    try:
        user_id = int(payload.get("user_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="user_id must be a valid integer")
    display_mode = str(payload.get("display_mode") or "athlete")
    report = payload.get("report") in (True, "true", "True", "1", 1)
    rotation_override: Optional[int] = None
    rotation = payload.get("rotation")
    if rotation is not None and str(rotation).strip() not in ("", "null", "undefined", "none"):
        try:
            rotation_override = int(rotation)
        except (ValueError, TypeError):
            rotation_override = None

    read_db = SessionLocal()
    try:
        db_user = person_crud.get_person(read_db, user_id)
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        user_name = db_user.name
        attempt_number = video_crud.get_next_attempt_number(read_db, user_id)
    finally:
        close_session_quietly(read_db)

    job_id = str(uuid.uuid4())
    file_name = os.path.basename(s3_key)
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
    queue_job(
        background_tasks,
        job_id,
        "s3",
        user_id,
        str(payload.get("file_name") or file_name),
        display_mode,
        report,
        {
            "s3_key": s3_key,
            "file_path": os.path.join(TEMP_VIDEO_DIR, f"{job_id}_{file_name}"),
            "user_id": user_id,
            "display_mode": display_mode,
            "report": report,
            "user_name": user_name,
            "attempt_number": attempt_number,
            "rotation_override": rotation_override,
        },
    )
    return {"job_id": job_id, "status": "pending", "message": "Video upload successful. Processing started in background."}
