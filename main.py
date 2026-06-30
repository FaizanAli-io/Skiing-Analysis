from fastapi import FastAPI, UploadFile, File, Form, Depends, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import shutil
import os
from datetime import date

# Local imports
from database import Base, engine, ensure_database_schema, get_db, SessionLocal
from models.person import Person
from models.video_analysis import VideoAnalysis
from schemas.person import PersonIDName
import crud.person as person_crud
import crud.video_analysis as video_crud
from services.new_analysis import analyze_video
from routes import admin_portal, app_routes, auth, client_portal, person, users, video_analysis
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

# Include route modules
app.include_router(video_analysis.router)
app.include_router(person.router)
app.include_router(users.router)
app.include_router(auth.router)
app.include_router(admin_portal.router)
app.include_router(client_portal.router)
app.include_router(app_routes.router)


# @app.on_event("startup")
# def start_background_tasks():
#     watcher_thread = threading.Thread(target=start_watching, daemon=True)
#     watcher_thread.start()


# Custom endpoint: Get ID and Name of all persons
@app.get("/all", response_model=List[PersonIDName])
def get_all_persons_id_name(
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return person_crud.get_all_persons_id_name(db)

# Video upload and analysis endpoint
@app.post("/analyze/")
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


@app.post("/analyze-react-overlay/")
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


@app.post("/analyze-premium-overlay/")
async def analyze_ski_video_premium_overlay(
    user_id: int = Form(...),
    display_mode: str = Form("athlete"),
    report: bool = Form(False),
    file: UploadFile = File(...),
    current_admin: Person = Depends(require_admin),
):
    read_db = SessionLocal()
    try:
        db_user = person_crud.get_person(read_db, user_id)
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        user_name = db_user.name
        attempt_number = video_crud.get_next_attempt_number(read_db, user_id)
    finally:
        read_db.close()

    session_date = date.today().isoformat()
    os.makedirs("temp_videos", exist_ok=True)
    file_location = f"temp_videos/{file.filename}"

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file.file.close()

    results = analyze_video(
        file_location,
        display_mode=display_mode,
        overlay_renderer="premium",
        report=report,
        user_name=user_name,
        attempt_number=attempt_number,
        session_date=session_date,
    )
    
    # Upload to S3 if enabled
    s3_video_url = None
    s3_report_url = None
    s3_snapshot_url = None
    s3_uploads = {}
    
    if S3Manager.is_enabled():
        s3_uploads = upload_analysis_files(
            video_path=results.get("output_path"),
            report_path=results.get("report_path"),
            snapshot_path=results.get("snapshot_path")
        )
        s3_video_url = s3_uploads.get("video_url")
        s3_report_url = s3_uploads.get("report_url")
        s3_snapshot_url = s3_uploads.get("snapshot_url")
        
        # Use S3 URLs if available
        if s3_video_url:
            results["video_url"] = s3_video_url
        if s3_report_url:
            results["report_url"] = s3_report_url
    
    # Fallback to local URLs if S3 not enabled
    if not s3_video_url and "output_path" in results:
        results["video_url"] = f"/outputs/{os.path.basename(results['output_path'])}"
    if not s3_report_url and "report_path" in results:
        results["report_url"] = f"/outputs/{os.path.basename(results['report_path'])}"
    
    blue_iq_score = (
        results["pressure_score"]
        + results["balance_score"]
        + results["rotation_score"]
        + results["edging_score"]
    ) / 4
    write_db = SessionLocal()
    try:
        attempt = VideoAnalysis(
            person_id=user_id,
            attempt_number=attempt_number,
            video_name=file.filename,
            video_link=results.get("video_url"),  # S3 presigned URL or local URL
            input_video_path=file_location,
            output_video_path=results.get("output_path"),  # Local path
            report_path=s3_report_url if s3_report_url else results.get("report_path"),  # S3 URL or local path
            s3_video_key=s3_uploads.get("video_s3_key") if S3Manager.is_enabled() else None,
            s3_report_key=s3_uploads.get("report_s3_key") if S3Manager.is_enabled() else None,
            s3_snapshot_key=s3_uploads.get("snapshot_s3_key") if S3Manager.is_enabled() else None,
            display_mode=display_mode,
            overlay_renderer="premium",
            blue_iq_score=blue_iq_score,
            pressure_score=results["pressure_score"],
            balance_score=results["balance_score"],
            rotation_score=results["rotation_score"],
            edging_score=results["edging_score"],
            turns=results.get("turns"),
            duration=results.get("duration"),
            status="completed",
        )
        write_db.add(attempt)
        write_db.commit()
        write_db.refresh(attempt)
        results["attempt_id"] = attempt.id
    finally:
        write_db.close()

    results["attempt_number"] = attempt_number
    results["session_date"] = session_date
    results["user_id"] = user_id
    results["user_name"] = user_name

    return results
