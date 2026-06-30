from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime,date
from fastapi import Form

from models.person import Person
from models.video_analysis import VideoAnalysis
from database import get_db
from pydantic import BaseModel
from services.auth import require_admin
router = APIRouter(prefix="/app", tags=["App-Data"])


# Request & Response Schemas
class AppRequest(BaseModel):
    before: Optional[datetime] = None
    after: Optional[datetime] = None

class Score(BaseModel):
    date: datetime
    metric_1: float
    metric_2: float
    metric_3: float
    metric_4: float

class PersonScores(BaseModel):
    name: str
    email: str
    scores: List[Score]


from fastapi import Form

@router.post("/app", response_model=List[PersonScores])
def get_app_data(
    before: Optional[date] = Form(None),
    after: Optional[date] = Form(None),
    _admin: Person = Depends(require_admin),
    db: Session = Depends(get_db)
):
    if before is None and after is None:
        raise HTTPException(status_code=400, detail="Either 'before' or 'after' must be provided.")

    query = db.query(Person).join(VideoAnalysis).distinct()
    result = []

    for person in query.all():
        videos = db.query(VideoAnalysis).filter(VideoAnalysis.person_id == person.id).all()

        # Filter based on date only (not time)
        filtered_videos = []
        for video in videos:
            if not video.timestamp:
                    continue
             
            video_date = video.timestamp.date()
            if before and after:
                if after <= video_date <= before:
                    filtered_videos.append(video)
            elif before:
                if video_date <= before:
                    filtered_videos.append(video)
            elif after:
                if video_date >= after:
                    filtered_videos.append(video)

        if not filtered_videos:
            continue

        filtered_videos.sort(key=lambda v: v.timestamp, reverse=True)

        scores = [
            Score(
                date=video.timestamp,
                metric_1=video.pressure_score,
                metric_2=video.balance_score,
                metric_3=video.rotation_score,
                metric_4=video.edging_score
            ) for video in filtered_videos
        ]

        result.append(PersonScores(name=person.name, email=person.email, scores=scores))

    return result

