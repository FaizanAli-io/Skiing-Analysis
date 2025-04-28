from fastapi import FastAPI, UploadFile, File,Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from services.analysis import analyze_video
from database import Base, engine
from routes import video_analysis, person,app_routes
from models.video_analysis import VideoAnalysis

from database import get_db
# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Ski Video Analyzer",
    description="API for analyzing skiing videos and managing persons",
    version="1.0.0"
)

# CORS middleware (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(video_analysis.router)
app.include_router(person.router)
app.include_router(app_routes.router)
# Video analysis endpoint
@app.post("/analyze/")
async def analyze_ski_video(person_id: int = Form(...),file: UploadFile = File(...)):
    file_location = f"temp_videos/{file.filename}"
    os.makedirs("temp_videos", exist_ok=True)

    # Save uploaded file
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file.file.close()

    # Run analysis
    video_path = file_location
    results = analyze_video(file_location)
    new_entry = VideoAnalysis(
        person_id=person_id,
        video_name=file.filename,
        pressure_score=results["pressure_score"],
        balance_score=results["balance_score"],
        rotation_score=results["rotation_score"],
        edging_score=results["edging_score"]
    )
    db = next(get_db())
    db.add(new_entry)
    db.commit()
    # Optionally delete file after analysis
    #os.remove(file_location)

    return results
