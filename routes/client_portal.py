from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from schemas.video_analysis import VideoAnalysisOut
from services.auth import get_current_user
from services.aws_s3 import S3Manager


router = APIRouter(prefix="/me", tags=["Client Portal"])


@router.get("/attempts", response_model=List[VideoAnalysisOut])
def my_attempts(
    skip: int = 0,
    limit: int = 50,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempts = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.person_id == current_user.id)
        .order_by(VideoAnalysis.attempt_number.desc().nullslast(), VideoAnalysis.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    # Refresh S3 presigned URLs if S3 is enabled
    if S3Manager.is_enabled():
        for attempt in attempts:
            if attempt.s3_video_key:
                attempt.video_link = S3Manager.get_video_url(attempt.s3_video_key, expiration=86400)
            if attempt.s3_report_key:
                attempt.report_path = S3Manager.get_report_url(attempt.s3_report_key, expiration=86400)
    
    return attempts


@router.get("/attempts/{attempt_id}", response_model=VideoAnalysisOut)
def my_attempt_detail(
    attempt_id: int,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempt = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.id == attempt_id, VideoAnalysis.person_id == current_user.id)
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return attempt


@router.get("/trends")
def get_performance_trends(
    limit: int = 15,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get performance trends over multiple runs for charts and analysis"""
    
    attempts = (
        db.query(VideoAnalysis)
        .filter(VideoAnalysis.person_id == current_user.id)
        .order_by(VideoAnalysis.created_at.desc())
        .limit(limit)
        .all()
    )
    
    if len(attempts) < 2:
        return {"error": "Need at least 2 runs for trend analysis", "total_runs": len(attempts)}
    
    # Reverse to get oldest first for trend calculation
    attempts = list(reversed(attempts))
    
    # Extract data for each metric
    dates = [a.created_at.strftime("%Y-%m-%d") for a in attempts]
    blue_iq = [round(a.blue_iq_score or 0, 1) for a in attempts]
    balance = [round(a.balance_score or 0, 1) for a in attempts]
    rotation = [round(a.rotation_score or 0, 1) for a in attempts]
    pressure = [round(a.pressure_score or 0, 1) for a in attempts]
    edging = [round(a.edging_score or 0, 1) for a in attempts]
    turns = [a.turns or 0 for a in attempts]
    
    def calculate_trend(values):
        """Calculate if trend is improving, declining, or stable using linear regression"""
        if len(values) < 2:
            return "stable"
        
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 1:
            return "improving"
        elif slope < -1:
            return "declining"
        else:
            return "stable"
    
    def calculate_improvement_rate(values):
        """Calculate average improvement percentage per run"""
        if len(values) < 2:
            return 0
        first = values[0]
        last = values[-1]
        runs = len(values) - 1
        if first == 0:
            return 0
        return round(((last - first) / first / runs) * 100, 2)
    
    def find_best_worst(values, dates_list):
        """Find best and worst run indices"""
        if not values:
            return None
        best_idx = values.index(max(values))
        worst_idx = values.index(min(values))
        return {
            "best": {"date": dates_list[best_idx], "score": values[best_idx], "run": best_idx + 1},
            "worst": {"date": dates_list[worst_idx], "score": values[worst_idx], "run": worst_idx + 1}
        }
    
    def calc_change(first_val, last_val):
        """Calculate change and percentage"""
        change = round(last_val - first_val, 1)
        change_percent = round(((last_val - first_val) / first_val * 100), 1) if first_val > 0 else 0
        return {"change": change, "change_percent": change_percent}
    
    return {
        "total_runs": len(attempts),
        "date_range": {
            "start": dates[0],
            "end": dates[-1]
        },
        "time_series": {
            "dates": dates,
            "blue_iq": blue_iq,
            "balance": balance,
            "rotation": rotation,
            "pressure": pressure,
            "edging": edging,
            "turns": turns
        },
        "trends": {
            "blue_iq": calculate_trend(blue_iq),
            "balance": calculate_trend(balance),
            "rotation": calculate_trend(rotation),
            "pressure": calculate_trend(pressure),
            "edging": calculate_trend(edging)
        },
        "improvement_rates": {
            "blue_iq": calculate_improvement_rate(blue_iq),
            "balance": calculate_improvement_rate(balance),
            "rotation": calculate_improvement_rate(rotation),
            "pressure": calculate_improvement_rate(pressure),
            "edging": calculate_improvement_rate(edging)
        },
        "highlights": {
            "blue_iq": find_best_worst(blue_iq, dates),
            "balance": find_best_worst(balance, dates),
            "rotation": find_best_worst(rotation, dates),
            "pressure": find_best_worst(pressure, dates),
            "edging": find_best_worst(edging, dates)
        },
        "current_vs_first": {
            "blue_iq": {
                "first": blue_iq[0],
                "current": blue_iq[-1],
                **calc_change(blue_iq[0], blue_iq[-1])
            },
            "balance": {
                "first": balance[0],
                "current": balance[-1],
                **calc_change(balance[0], balance[-1])
            },
            "rotation": {
                "first": rotation[0],
                "current": rotation[-1],
                **calc_change(rotation[0], rotation[-1])
            },
            "pressure": {
                "first": pressure[0],
                "current": pressure[-1],
                **calc_change(pressure[0], pressure[-1])
            },
            "edging": {
                "first": edging[0],
                "current": edging[-1],
                **calc_change(edging[0], edging[-1])
            }
        }
    }

