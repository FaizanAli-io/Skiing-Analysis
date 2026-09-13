from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class JobStatus(BaseModel):
    job_id: str
    person_id: int
    person_name: Optional[str] = None
    file_name: str
    display_mode: Optional[str] = "coach"
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    video_analysis_id: Optional[int] = None
    
    class Config:
        from_attributes = True


class JobStatusWithResult(JobStatus):
    result: Optional[Any] = None  # VideoAnalysis data when completed
    
    class Config:
        from_attributes = True


class JobCreate(BaseModel):
    person_id: int
    file_name: str
    display_mode: str = "coach"
    generate_report: bool = True
