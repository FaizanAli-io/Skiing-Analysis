from pydantic import BaseModel, Field, field_validator
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
    instructor_notes: Optional[str] = None
    is_archived: Optional[bool] = False
    timestamp: Optional[datetime] = None

    @field_validator("is_archived", mode="before")
    @classmethod
    def coerce_archived_create(cls, v):
        return bool(v) if v is not None else False

class VideoAnalysisOut(VideoAnalysisCreate):
    id: int
    is_archived: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    session_date: Optional[str] = None
    session_number: Optional[int] = None
    run_number: Optional[int] = None
    person_name: Optional[str] = None
    client_name: Optional[str] = None
    personal_best_comparisons: Dict[str, ScoreComparison] = Field(default_factory=dict)
    new_personal_bests: List[str] = Field(default_factory=list)
    notes_source: Optional[str] = None
    available_downloads: Dict[str, List[str]] = Field(default_factory=dict)
    downloads_pending: bool = False

    @field_validator("is_archived", mode="before")
    @classmethod
    def coerce_archived_out(cls, v):
        return bool(v) if v is not None else False

    class Config:
        from_attributes = True


class ClientAttemptOut(BaseModel):
    """What a client may see about their own run.

    Deliberately excludes storage paths, S3 keys, internal file names, render
    settings, archive state and note provenance.
    """

    id: int
    video_link: Optional[str] = None
    report_path: Optional[str] = None
    blue_iq_score: Optional[float] = None
    pressure_score: float
    balance_score: float
    rotation_score: float
    edging_score: float
    turns: Optional[int] = None
    duration: Optional[float] = None
    status: Optional[str] = "completed"
    instructor_notes: Optional[str] = None
    timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    session_date: Optional[str] = None
    session_number: Optional[int] = None
    run_number: Optional[int] = None
    person_name: Optional[str] = None
    client_name: Optional[str] = None
    personal_best_comparisons: Dict[str, ScoreComparison] = Field(default_factory=dict)
    new_personal_bests: List[str] = Field(default_factory=list)
    available_downloads: Dict[str, List[str]] = Field(default_factory=dict)
    downloads_pending: bool = False

    class Config:
        from_attributes = True
