from fastapi import FastAPI, UploadFile, File, Form, Depends, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import shutil
import os
import re
from datetime import date
import uuid

# Local imports
from database import (
    Base,
    SessionLocal,
    close_session_quietly,
    engine,
    ensure_database_schema,
    get_db,
)
from models.person import Person
from models.video_analysis import VideoAnalysis
from models.analysis_timeline import AnalysisTimeline
from schemas.person import PersonIDName
import crud.person as person_crud
import crud.video_analysis as video_crud
import crud.job as job_crud
from services.new_analysis import analyze_video
from services.background_tasks import process_video_analysis_background
from routes import admin_portal, app_routes, auth, client_portal, person, users, video_analysis, jobs
import threading
from services.file_watcher import start_watching
from typing import Optional, List
from services.auth import hash_password, require_admin
from services.aws_s3 import S3Manager, upload_analysis_files


# Create DB tables
Base.metadata.create_all(bind=engine)
ensure_database_schema()


def ensure_default_admin():
    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return

    db = SessionLocal()
    try:
        email = admin_email.strip().lower()
        existing = db.query(Person).filter(Person.email == email).first()
        if existing:
            if existing.role != "admin":
                existing.role = "admin"
                db.commit()
            return
        admin = Person(
            name=os.getenv("DEFAULT_ADMIN_NAME", "Bluerun Admin"),
            email=email,
            password_hash=hash_password(admin_password),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
    finally:
        db.close()


ensure_default_admin()

# FastAPI app instance
app = FastAPI(
    title="Ski Video Analyzer",
    description="API for analyzing skiing videos and managing persons",
    version="1.0.0"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Include route modules with /api prefix
app.include_router(video_analysis.router, prefix="/api")
app.include_router(person.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(admin_portal.router, prefix="/api")
app.include_router(client_portal.router, prefix="/api")
app.include_router(app_routes.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")


# @app.on_event("startup")
# def start_background_tasks():
#     watcher_thread = threading.Thread(target=start_watching, daemon=True)
#     watcher_thread.start()


# Custom endpoint: Get ID and Name of all persons
@app.get("/api/all", response_model=List[PersonIDName])
def get_all_persons_id_name(
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return person_crud.get_all_persons_id_name(db)

# Video upload and analysis endpoint
@app.post("/api/analyze/")
async def analyze_ski_video(
    person_id: Optional[str] = Form(None),
    display_mode: str = Form("coach"),
    file: UploadFile = File(...),
    current_admin: Person = Depends(require_admin),
):
    os.makedirs("temp_videos", exist_ok=True)
    file_location = f"temp_videos/{file.filename}"

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file.file.close()

    normalized_person_id = None
    if person_id not in (None, ""):
        try:
            normalized_person_id = int(person_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="person_id must be a valid integer")

    results = analyze_video(file_location, display_mode=display_mode)

    db = next(get_db())
    # new_entry = VideoAnalysis(
    #     person_id=person_id,
    #     video_name=file.filename,
    #     pressure_score=results["pressure_score"],
    #     balance_score=results["balance_score"],
    #     rotation_score=results["rotation_score"],
    #     edging_score=results["edging_score"]
    # )
    new_entry = VideoAnalysis(
    video_name=file.filename,
    pressure_score=results["pressure_score"],
    balance_score=results["balance_score"],
    rotation_score=results["rotation_score"],
    edging_score=results["edging_score"]    
    )

    if normalized_person_id is not None:
        new_entry.person_id = normalized_person_id

        db.add(new_entry)
        db.commit()

    # Optionally delete file
    # os.remove(file_location)

    return results


@app.post("/api/analyze-react-overlay/")
async def analyze_ski_video_react_overlay(
    display_mode: str = Form("athlete"),
    file: UploadFile = File(...),
    current_admin: Person = Depends(require_admin),
):
    os.makedirs("temp_videos", exist_ok=True)
    file_location = f"temp_videos/{file.filename}"

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file.file.close()

    results = analyze_video(file_location, display_mode=display_mode, overlay_renderer="react")
    if "react_overlay_path" in results:
        results["react_overlay_url"] = f"/outputs/{os.path.basename(results['react_overlay_path'])}"
    if "output_path" in results:
        results["video_url"] = f"/outputs/{os.path.basename(results['output_path'])}"

    return results


@app.post("/api/analyze-premium-overlay/")
async def analyze_ski_video_premium_overlay(
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    display_mode: str = Form("athlete"),
    report: bool = Form(False),
    file: UploadFile = File(...),
    current_admin: Person = Depends(require_admin),
):
    """
    Upload a video for background processing.
    Returns immediately with a job_id to track progress.
    """
    # Validate user exists
    read_db = SessionLocal()
    try:
        db_user = person_crud.get_person(read_db, user_id)
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        user_name = db_user.name
        attempt_number = video_crud.get_next_attempt_number(read_db, user_id)
    finally:
        close_session_quietly(read_db)

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    os.makedirs("temp_videos", exist_ok=True)
    file_location = f"temp_videos/{job_id}_{file.filename}"
    
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file.file.close()
    
    # Create job record in database
    write_db = SessionLocal()
    try:
        job_crud.create_job(
            write_db,
            job_id=job_id,
            person_id=user_id,
            file_name=file.filename,
            display_mode=display_mode,
            generate_report=report
        )
    finally:
        close_session_quietly(write_db)
    
    # Add background task
    background_tasks.add_task(
        process_video_analysis_background,
        job_id=job_id,
        file_path=file_location,
        user_id=user_id,
        display_mode=display_mode,
        report=report,
        user_name=user_name,
        attempt_number=attempt_number
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Video upload successful. Processing started in background."
    }


@app.get("/api/config/google-drive")
def get_google_drive_config():
    """Return public Google API keys / client ID for Google Picker."""
    return {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "app_id": os.getenv("GOOGLE_APP_ID", ""),
    }


@app.post("/api/analyze-google-drive/")
async def analyze_ski_video_google_drive(
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    display_mode: str = Form("coach"),
    report: bool = Form(False),
    drive_file_id: Optional[str] = Form(None),
    drive_url: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None),
    oauth_token: Optional[str] = Form(None),
    current_admin: Person = Depends(require_admin),
):
    """
    Queue a Google Drive video for background analysis.
    Downloads the video directly in the worker and runs the sequential pipeline.
    """
    from services.google_drive import extract_drive_file_id, get_drive_file_metadata
    from services.background_tasks import process_google_drive_analysis_background

    resolved_file_id = extract_drive_file_id(drive_file_id or drive_url or "")
    if not resolved_file_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid Google Drive link or File ID. Please check the URL and try again."
        )

    # Validate user exists
    read_db = SessionLocal()
    try:
        db_user = person_crud.get_person(read_db, user_id)
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        user_name = db_user.name
        attempt_number = video_crud.get_next_attempt_number(read_db, user_id)
    finally:
        close_session_quietly(read_db)

    # Resolve filename
    actual_filename = file_name
    if not actual_filename or actual_filename.strip() in ("", "undefined"):
        meta = get_drive_file_metadata(resolved_file_id, oauth_token=oauth_token)
        actual_filename = meta.get("name", f"gdrive_{resolved_file_id[:8]}.mp4")

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    os.makedirs("temp_videos", exist_ok=True)
    clean_name = re.sub(r'[^\w\-_\.]', '_', actual_filename)
    if not clean_name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        clean_name += ".mp4"
    file_location = f"temp_videos/{job_id}_{clean_name}"

    # Create job record in database
    write_db = SessionLocal()
    try:
        job_crud.create_job(
            write_db,
            job_id=job_id,
            person_id=user_id,
            file_name=actual_filename,
            display_mode=display_mode,
            generate_report=report
        )
    finally:
        close_session_quietly(write_db)

    # Add background task for sequential queue
    background_tasks.add_task(
        process_google_drive_analysis_background,
        job_id=job_id,
        drive_file_id=resolved_file_id,
        file_path=file_location,
        user_id=user_id,
        display_mode=display_mode,
        report=report,
        user_name=user_name,
        attempt_number=attempt_number,
        oauth_token=oauth_token,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "file_name": actual_filename,
        "message": f"Google Drive video '{actual_filename}' queued for analysis."
    }
