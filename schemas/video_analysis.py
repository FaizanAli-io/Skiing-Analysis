from pydantic import BaseModel
from typing import Optional
from datetime import datetime
class VideoAnalysisCreate(BaseModel):
    person_id: int
    video_name: str
    video_link: Optional[str]
    pressure_score: float
    balance_score: float
    rotation_score: float
    edging_score: float
    timestamp: datetime 

class VideoAnalysisOut(VideoAnalysisCreate):
    id: int

    class Config:
        from_attributes = True