from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import crud.video_analysis as crud
import schemas.video_analysis as schemas

from typing import List, Optional
from datetime import datetime
import secrets

from models.person import Person
from models.video_analysis import VideoAnalysis
from database import get_db
from pydantic import BaseModel


router = APIRouter(prefix="/videos", tags=["Video Analysis"])

@router.post("/", response_model=schemas.VideoAnalysisOut)
def create(video: schemas.VideoAnalysisCreate, db: Session = Depends(get_db)):
    return crud.create_video_analysis(db, video)

@router.get("/{video_id}", response_model=schemas.VideoAnalysisOut)
def read(video_id: int, db: Session = Depends(get_db)):
    db_video = crud.get_video(db, video_id)
    if db_video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return db_video

@router.get("/", response_model=list[schemas.VideoAnalysisOut])
def read_all(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.get_all_videos(db, skip, limit)

@router.put("/{video_id}", response_model=schemas.VideoAnalysisOut)
def update(video_id: int, video: schemas.VideoAnalysisCreate, db: Session = Depends(get_db)):
    updated = crud.update_video(db, video_id, video)
    if not updated:
        raise HTTPException(status_code=404, detail="Video not found")
    return updated

@router.delete("/{video_id}")
def delete(video_id: int, db: Session = Depends(get_db)):
    deleted = crud.delete_video(db, video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"message": "Deleted successfully"}




