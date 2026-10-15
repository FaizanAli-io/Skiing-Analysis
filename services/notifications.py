"""Creating notifications when runs finish (or fail)."""

import logging
import math
from typing import Optional

from database import SessionLocal, close_session_quietly

logger = logging.getLogger(__name__)


def _blue_iq(run) -> Optional[int]:
    if run.blue_iq_score is not None:
        return int(math.ceil(run.blue_iq_score))
    scores = [run.pressure_score, run.balance_score, run.rotation_score, run.edging_score]
    if all(score is not None for score in scores):
        return int(math.ceil(sum(scores) / 4))
    return None


def notify_run_ready(analysis_id: int) -> None:
    """Tell the athlete and every admin that a run has finished processing."""
    from models.notification import Notification
    from models.person import Person
    from models.video_analysis import VideoAnalysis
    from services.personal_bests import client_run_number

    db = SessionLocal()
    try:
        run = db.query(VideoAnalysis).filter(VideoAnalysis.id == analysis_id).first()
        if run is None:
            return
        athlete = db.query(Person).filter(Person.id == run.person_id).first()
        name = athlete.name if athlete else "An athlete"
        first_name = name.split(" ")[0]
        score = _blue_iq(run)
        score_text = f" · Blue IQ {score}" if score is not None else ""

        if athlete and athlete.role != "admin":
            client_number = client_run_number(db, run) or run.attempt_number or run.id
            db.add(Notification(
                person_id=athlete.id,
                kind="run_ready",
                emoji="🎉",
                title="Your new run is ready!",
                body=f"Run {client_number}{score_text}. Tap to download your video.",
                analysis_id=run.id,
                athlete_id=athlete.id,
            ))

        for admin in db.query(Person).filter(Person.role == "admin", Person.is_active.is_(True)).all():
            db.add(Notification(
                person_id=admin.id,
                kind="run_ready",
                emoji="✅",
                title=f"{first_name}'s run {run.attempt_number or run.id} is ready",
                body=f"{name}{score_text}",
                analysis_id=run.id,
                athlete_id=run.person_id,
            ))
        db.commit()
    except Exception as exc:
        logger.warning("Could not create notifications for analysis %s: %s", analysis_id, exc)
        db.rollback()
    finally:
        close_session_quietly(db)


def notify_run_failed(job_id: str) -> None:
    """Tell admins that an upload could not be processed."""
    from models.analysis_job import AnalysisJob
    from models.notification import Notification
    from models.person import Person

    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job is None:
            return
        athlete = db.query(Person).filter(Person.id == job.person_id).first()
        name = athlete.name if athlete else "an athlete"
        for admin in db.query(Person).filter(Person.role == "admin", Person.is_active.is_(True)).all():
            db.add(Notification(
                person_id=admin.id,
                kind="run_failed",
                emoji="⚠️",
                title=f"Upload for {name} failed",
                body=f"{job.file_name} could not be processed. Please upload it again.",
                athlete_id=job.person_id,
            ))
        db.commit()
    except Exception as exc:
        logger.warning("Could not create failure notification for job %s: %s", job_id, exc)
        db.rollback()
    finally:
        close_session_quietly(db)


MODE_LABELS = {"athlete": "Athlete", "coach": "Coach"}
FORMAT_LABELS = {"original": "Original", "vertical": "Vertical", "square": "Square"}


def version_label(mode: str, fmt: str) -> str:
    return f"{MODE_LABELS.get(mode, mode.title())} · {FORMAT_LABELS.get(fmt, fmt.title())}"


def notify_version_result(analysis_id: int, mode: str, fmt: str, person_ids, ok: bool) -> None:
    """Tell whoever asked for a download version that it is ready (or failed)."""
    from models.notification import Notification
    from models.video_analysis import VideoAnalysis

    db = SessionLocal()
    try:
        run = db.query(VideoAnalysis).filter(VideoAnalysis.id == analysis_id).first()
        if run is None:
            return
        label = version_label(mode, fmt)
        for person_id in dict.fromkeys(person_ids):
            db.add(Notification(
                person_id=person_id,
                kind="video_ready" if ok else "video_failed",
                emoji="🎬" if ok else "⚠️",
                title=f"Your {label} video is ready" if ok else f"Your {label} video could not be made",
                body="Tap to download it." if ok else "Please try again from the Download menu.",
                analysis_id=run.id,
                athlete_id=run.person_id,
                video_mode=mode,
                video_format=fmt,
            ))
        db.commit()
    except Exception as exc:
        logger.warning("Could not create version notification for analysis %s: %s", analysis_id, exc)
        db.rollback()
    finally:
        close_session_quietly(db)

