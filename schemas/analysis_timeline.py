from typing import Any, Dict, List

from pydantic import BaseModel


class AnalysisTimelineOut(BaseModel):
    analysis_id: int
    resolution: str
    duration_seconds: float
    sample_rate_hz: float
    scoring_version: str
    parameter_config: Dict[str, Any]
    samples: List[Dict[str, Any]]
