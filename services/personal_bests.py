"""Shared personal-best calculations for portals, reports, and leaderboards."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sqlalchemy import or_

from models.person import Person
from models.video_analysis import VideoAnalysis


METRIC_FIELDS = {
    "blue_iq": "blue_iq_score",
    "pressure": "pressure_score",
    "balance": "balance_score",
    "rotation": "rotation_score",
    "edging": "edging_score",
}

METRIC_LABELS = {
    "blue_iq": "Blue IQ",
    "pressure": "Pressure",
    "balance": "Balance",
    "rotation": "Rotation",
    "edging": "Edging",
}

METRIC_KEYS = tuple(METRIC_FIELDS)


def display_score(value: Any) -> Optional[int]:
    """Use the same ceiling rule as the video, report, and portals."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(0, min(240, int(math.ceil(numeric))))


def metric_score(attempt: Any, metric: str) -> Optional[int]:
    if metric not in METRIC_FIELDS:
        raise ValueError(f"Unknown score metric: {metric}")

    value = getattr(attempt, METRIC_FIELDS[metric], None)
    if metric == "blue_iq" and value is None:
        pillar_values = [
            getattr(attempt, METRIC_FIELDS[key], None)
            for key in ("pressure", "balance", "rotation", "edging")
        ]
        if all(value is not None for value in pillar_values):
            value = sum(float(value) for value in pillar_values) / 4.0
    return display_score(value)


def _attempt_datetime(attempt: Any) -> Optional[datetime]:
    value = getattr(attempt, "created_at", None) or getattr(
        attempt, "timestamp", None
    )
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if value:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None
    return None


def _date_key(attempt: Any) -> str:
    value = _attempt_datetime(attempt)
    return value.date().isoformat() if value else "Unknown date"


def _sort_key(attempt: Any) -> tuple:
    value = _attempt_datetime(attempt)
    date_key = value.isoformat() if value else "0001-01-01T00:00:00"
    return date_key, int(getattr(attempt, "id", 0) or 0)


def is_completed_attempt(attempt: Any) -> bool:
    status = getattr(attempt, "status", None)
    return status is None or str(status).strip().lower() == "completed"


def _session_numbers(attempts: Sequence[Any]) -> Dict[str, int]:
    date_keys = sorted({_date_key(attempt) for attempt in attempts})
    return {date_key: index + 1 for index, date_key in enumerate(date_keys)}


def _record(
    attempt: Any,
    metric: str,
    score: int,
    session_number: int,
) -> Dict[str, Any]:
    return {
        "metric": metric,
        "label": METRIC_LABELS[metric],
        "score": score,
        "analysis_id": int(getattr(attempt, "id", 0) or 0),
        "person_id": int(getattr(attempt, "person_id", 0) or 0),
        "date": _date_key(attempt),
        "session_number": session_number,
        "run_number": int(
            getattr(attempt, "attempt_number", None)
            or getattr(attempt, "id", 0)
            or 0
        ),
    }


def _comparison(
    metric: str,
    current_score: int,
    personal_best: Dict[str, Any],
    status: str,
    previous_best: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    difference = current_score - int(personal_best["score"])
    return {
        "metric": metric,
        "label": METRIC_LABELS[metric],
        "current_score": current_score,
        "personal_best_score": int(personal_best["score"]),
        "difference": difference,
        "points_below": max(0, -difference),
        "status": status,
        "is_new_personal_best": status == "new_personal_best",
        "personal_best": dict(personal_best),
        "previous_best": dict(previous_best) if previous_best else None,
    }


def build_personal_best_history(attempts: Iterable[Any]) -> Dict[str, Any]:
    """Build all-time records and per-run comparisons from valid attempts."""
    ordered = sorted(
        (attempt for attempt in attempts if is_completed_attempt(attempt)),
        key=_sort_key,
    )
    sessions = _session_numbers(ordered)
    running_records: Dict[str, Dict[str, Any]] = {}
    achievements: Dict[int, List[str]] = defaultdict(list)
    previous_by_attempt: Dict[int, Dict[str, Optional[Dict[str, Any]]]] = {}

    for attempt in ordered:
        attempt_id = int(getattr(attempt, "id", 0) or 0)
        attempt_previous: Dict[str, Optional[Dict[str, Any]]] = {}
        for metric in METRIC_KEYS:
            score = metric_score(attempt, metric)
            if score is None:
                continue
            previous = running_records.get(metric)
            attempt_previous[metric] = dict(previous) if previous else None
            current_record = _record(
                attempt,
                metric,
                score,
                sessions[_date_key(attempt)],
            )
            if previous is None:
                running_records[metric] = current_record
            elif score > int(previous["score"]):
                running_records[metric] = current_record
                achievements[attempt_id].append(metric)
        previous_by_attempt[attempt_id] = attempt_previous

    all_time_records = {
        metric: dict(record) for metric, record in running_records.items()
    }
    comparisons_by_attempt: Dict[int, Dict[str, Dict[str, Any]]] = {}
    metadata_by_attempt: Dict[int, Dict[str, Any]] = {}

    for attempt in ordered:
        attempt_id = int(getattr(attempt, "id", 0) or 0)
        date_key = _date_key(attempt)
        metadata_by_attempt[attempt_id] = {
            "session_date": date_key,
            "session_number": sessions[date_key],
            "run_number": int(
                getattr(attempt, "attempt_number", None) or attempt_id
            ),
        }
        comparisons: Dict[str, Dict[str, Any]] = {}
        for metric in all_time_records:
            score = metric_score(attempt, metric)
            if score is None:
                continue
            previous = previous_by_attempt.get(attempt_id, {}).get(metric)
            current_record = _record(
                attempt,
                metric,
                score,
                sessions[date_key],
            )
            if previous is None:
                status = "baseline"
                comparison_best = current_record
            elif metric in achievements.get(attempt_id, []):
                status = "new_personal_best"
                comparison_best = current_record
            elif score == int(previous["score"]):
                status = "matches_personal_best"
                comparison_best = previous
            else:
                status = "below_personal_best"
                comparison_best = previous
            comparisons[metric] = _comparison(
                metric,
                score,
                comparison_best,
                status,
                previous_best=previous,
            )
        comparisons_by_attempt[attempt_id] = comparisons

    return {
        "personal_bests": all_time_records,
        "comparisons_by_attempt": comparisons_by_attempt,
        "new_personal_bests_by_attempt": {
            attempt_id: list(metrics)
            for attempt_id, metrics in achievements.items()
        },
        "metadata_by_attempt": metadata_by_attempt,
    }


def compare_current_scores_to_previous(
    current_scores: Mapping[str, Any],
    previous_personal_bests: Optional[Mapping[str, Mapping[str, Any]]],
    *,
    person_id: int = 0,
    analysis_id: int = 0,
    session_date: Optional[str] = None,
    session_number: int = 0,
    run_number: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Compare an unsaved run with records that existed before processing."""
    previous_personal_bests = previous_personal_bests or {}
    comparisons: Dict[str, Dict[str, Any]] = {}
    for metric in METRIC_KEYS:
        current_score = display_score(current_scores.get(metric))
        if current_score is None:
            continue
        previous = previous_personal_bests.get(metric)
        current_record = {
            "metric": metric,
            "label": METRIC_LABELS[metric],
            "score": current_score,
            "analysis_id": analysis_id,
            "person_id": person_id,
            "date": session_date or date.today().isoformat(),
            "session_number": session_number,
            "run_number": run_number,
        }

        if previous is None:
            status = "baseline"
            personal_best = current_record
        elif current_score > int(previous["score"]):
            status = "new_personal_best"
            personal_best = current_record
        elif current_score == int(previous["score"]):
            status = "matches_personal_best"
            personal_best = dict(previous)
        else:
            status = "below_personal_best"
            personal_best = dict(previous)

        comparisons[metric] = _comparison(
            metric,
            current_score,
            personal_best,
            status,
            previous_best=dict(previous) if previous else None,
        )
    return comparisons


def _completed_attempts_query(db):
    return db.query(VideoAnalysis).filter(
        or_(
            VideoAnalysis.status == "completed",
            VideoAnalysis.status.is_(None),
        )
    )


def personal_bests_for_person(db, person_id: int) -> Dict[str, Dict[str, Any]]:
    attempts = (
        _completed_attempts_query(db)
        .filter(VideoAnalysis.person_id == person_id)
        .order_by(VideoAnalysis.created_at.asc().nullsfirst(), VideoAnalysis.id.asc())
        .all()
    )
    return build_personal_best_history(attempts)["personal_bests"]


def pre_run_personal_best_context(
    db,
    person_id: int,
    session_date: str,
) -> Dict[str, Any]:
    """Return records and the date-derived session number before a new run."""
    attempts = (
        _completed_attempts_query(db)
        .filter(VideoAnalysis.person_id == person_id)
        .order_by(VideoAnalysis.created_at.asc().nullsfirst(), VideoAnalysis.id.asc())
        .all()
    )
    history = build_personal_best_history(attempts)
    existing_dates = sorted({_date_key(attempt) for attempt in attempts})
    if session_date in existing_dates:
        session_number = existing_dates.index(session_date) + 1
    else:
        session_number = len(existing_dates) + 1
    return {
        "personal_bests": history["personal_bests"],
        "session_number": max(1, session_number),
    }


def enrich_attempts(db, attempts: Sequence[VideoAnalysis]) -> Sequence[VideoAnalysis]:
    """Attach computed PB fields consumed by the Pydantic response schema."""
    person_ids = sorted({attempt.person_id for attempt in attempts if attempt.person_id})
    if not person_ids:
        return attempts

    histories = (
        _completed_attempts_query(db)
        .filter(VideoAnalysis.person_id.in_(person_ids))
        .order_by(VideoAnalysis.created_at.asc().nullsfirst(), VideoAnalysis.id.asc())
        .all()
    )
    grouped: Dict[int, List[VideoAnalysis]] = defaultdict(list)
    for attempt in histories:
        grouped[int(attempt.person_id)].append(attempt)

    calculated = {
        person_id: build_personal_best_history(rows)
        for person_id, rows in grouped.items()
    }
    for attempt in attempts:
        history = calculated.get(int(attempt.person_id or 0), {})
        attempt_id = int(attempt.id)
        metadata = history.get("metadata_by_attempt", {}).get(attempt_id, {})
        setattr(
            attempt,
            "personal_best_comparisons",
            history.get("comparisons_by_attempt", {}).get(attempt_id, {}),
        )
        setattr(
            attempt,
            "new_personal_bests",
            history.get("new_personal_bests_by_attempt", {}).get(attempt_id, []),
        )
        setattr(attempt, "session_date", metadata.get("session_date"))
        setattr(attempt, "session_number", metadata.get("session_number"))
        setattr(attempt, "run_number", metadata.get("run_number"))
    return attempts


def build_leaderboards(db, limit: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    attempts = (
        _completed_attempts_query(db)
        .filter(VideoAnalysis.person_id.isnot(None))
        .order_by(VideoAnalysis.created_at.asc().nullsfirst(), VideoAnalysis.id.asc())
        .all()
    )
    grouped: Dict[int, List[VideoAnalysis]] = defaultdict(list)
    for attempt in attempts:
        grouped[int(attempt.person_id)].append(attempt)

    people = {
        int(person.id): person
        for person in db.query(Person).filter(Person.id.in_(list(grouped))).all()
    } if grouped else {}

    records_by_metric: Dict[str, List[Dict[str, Any]]] = {
        metric: [] for metric in METRIC_KEYS
    }
    for person_id, rows in grouped.items():
        person = people.get(person_id)
        if person is None or getattr(person, "role", "client") == "admin":
            continue
        history = build_personal_best_history(rows)
        for metric, record in history["personal_bests"].items():
            entry = dict(record)
            entry.update({
                "athlete_name": person.name,
                "athlete_email": person.email,
            })
            records_by_metric[metric].append(entry)

    leaderboards: Dict[str, List[Dict[str, Any]]] = {}
    for metric, records in records_by_metric.items():
        ordered = sorted(
            records,
            key=lambda record: (
                -int(record["score"]),
                record["date"],
                record["athlete_name"].lower(),
            ),
        )[: max(1, min(int(limit), 250))]
        previous_score = None
        rank = 0
        for index, record in enumerate(ordered, start=1):
            if record["score"] != previous_score:
                rank = index
                previous_score = record["score"]
            record["rank"] = rank
        leaderboards[metric] = ordered
    return leaderboards
