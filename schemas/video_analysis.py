from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

from schemas.personal_best import ScoreComparison


class VideoAnalysisCreate(BaseModel):
    person_id: Optional[int] = None
    attempt_number: Optional[int] = None
    video_name: str
    video_link: Optional[str] = None
    input_video_path: Optional[str] = None
    output_video_path: Optional[str] = None
    report_path: Optional[str] = None
    display_mode: Optional[str] = None
    overlay_renderer: Optional[str] = None
    blue_iq_score: Optional[float] = None
    pressure_score: float
    balance_score: float
    rotation_score: float
    edging_score: float
    turns: Optional[int] = None
    duration: Optional[float] = None
    status: Optional[str] = "completed"
    timestamp: Optional[datetime] = None

class VideoAnalysisOut(VideoAnalysisCreate):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    session_date: Optional[str] = None
    session_number: Optional[int] = None
    run_number: Optional[int] = None
    personal_best_comparisons: Dict[str, ScoreComparison] = Field(default_factory=dict)
    new_personal_bests: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True
