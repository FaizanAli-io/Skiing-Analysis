from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


MetricKey = Literal["blue_iq", "pressure", "balance", "rotation", "edging"]


class PersonalBestRecord(BaseModel):
    metric: MetricKey
    label: str
    score: int
    analysis_id: int
    person_id: int
    date: str
    session_number: int
    run_number: int


class ScoreComparison(BaseModel):
    metric: MetricKey
    label: str
    current_score: int
    personal_best_score: int
    difference: int
    points_below: int
    status: Literal[
        "baseline",
        "new_personal_best",
        "matches_personal_best",
        "below_personal_best",
    ]
    is_new_personal_best: bool
    personal_best: PersonalBestRecord
    previous_best: Optional[PersonalBestRecord] = None


class PersonalBestsResponse(BaseModel):
    person_id: int
    personal_bests: Dict[str, PersonalBestRecord]


class LeaderboardEntry(PersonalBestRecord):
    rank: int
    athlete_name: str
    athlete_email: Optional[str] = None


class LeaderboardsResponse(BaseModel):
    leaderboards: Dict[str, List[LeaderboardEntry]]
