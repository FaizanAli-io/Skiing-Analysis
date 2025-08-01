from fastapi import FastAPI, UploadFile, File, Form, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os

# Local imports
from database import Base, engine, get_db, SessionLocal
from models.person import Person
from models.video_analysis import VideoAnalysis
from schemas.person import PersonIDName
import crud.person as person_crud
import crud.video_analysis as video_crud
from services.new_analysis import analyze_video
from routes import video_analysis, person, app_routes
import threading
from services.file_watcher import start_watching
from typing import Optional, List


# Create DB tables
Base.metadata.create_all(bind=engine)

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

# Include route modules
app.include_router(video_analysis.router)
app.include_router(person.router)
app.include_router(app_routes.router)


@app.on_event("startup")
def start_background_tasks():
    watcher_thread = threading.Thread(target=start_watching, daemon=True)
    watcher_thread.start()


persons = person_crud.get_all_persons_id_name(SessionLocal())

def format_persons(persons):
    formatted = []
    for person_id, name in persons:
        formatted.append(f"ID:{person_id} Name:{name}")
    return formatted

persons = format_persons(persons)

# Custom endpoint: Get ID and Name of all persons
@app.get("/all", response_model=List[PersonIDName])
def get_all_persons_id_name(db: Session = Depends(get_db)):
    return person_crud.get_all_persons_id_name(db)

# Video upload and analysis endpoint
@app.post("/analyze/")
async def analyze_ski_video(person_id: Optional[int] = Form(None), file: UploadFile = File(...)):
    os.makedirs("temp_videos", exist_ok=True)
    file_location = f"temp_videos/{file.filename}"

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file.file.close()

    results = analyze_video(file_location)

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

    if person_id is not None:
        new_entry.person_id = person_id

        db.add(new_entry)
        db.commit()

    # Optionally delete file
    # os.remove(file_location)

    return results
