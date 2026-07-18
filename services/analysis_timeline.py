"""Structured metric timelines for coach-facing run analysis charts."""

from collections import defaultdict
from copy import deepcopy
from math import floor, isfinite
from typing import Any, Dict, Iterable, List, Optional


SCORING_VERSION = "blueiq-2026-07-18-v1"


TIMELINE_PARAMETER_CONFIG: Dict[str, Any] = {
    "pressure": {
        "label": "Pressure",
        "color": "#22c55e",
        "score_series": [
            {
                "key": "overall",
                "label": "Pressure score",
                "path": "pillars.pressure.score",
                "color": "#eef6ff",
                "unit": "/240",
                "default_visible": True,
            },
            {
                "key": "vertical_range",
                "label": "Vertical movement",
                "path": "pillars.pressure.components.vertical_range.score",
                "color": "#22c55e",
                "unit": "/240",
                "weight": 75,
                "default_visible": True,
            },
            {
                "key": "knee_flexion_range",
                "label": "Knee flexion range",
                "path": "pillars.pressure.components.knee_flexion_range.score",
                "color": "#6aafff",
                "unit": "/240",
                "weight": 25,
                "default_visible": True,
            },
        ],
        "value_series": [
            {
                "key": "vertical_range",
                "label": "Vertical movement",
                "path": "pillars.pressure.components.vertical_range.value",
                "color": "#22c55e",
                "unit": "% body height",
                "axis": "percent",
                "default_visible": True,
            },
            {
                "key": "knee_flexion_range",
                "label": "Knee flexion range",
                "path": "pillars.pressure.components.knee_flexion_range.value",
                "color": "#6aafff",
                "unit": "deg",
                "axis": "degrees",
                "default_visible": True,
            },
        ],
        "references": [
            {
                "label": "Vertical movement",
                "explanation": (
                    "Across the analyzed run, this tracks how far the pelvis "
                    "rises and lowers relative to the skier's body height. A "
                    "larger up-and-down range earns a higher score because it "
                    "shows more active pressure release and reapplication."
                ),
                "mapping": "5% body-height range or less = 60; 12% or more = 240",
                "weight": "75% of Pressure",
            },
            {
                "label": "Knee flexion range",
                "explanation": (
                    "As the skier moves from one side of the run to the other, "
                    "this compares the more extended knee position with the "
                    "more flexed position. A change from 18 deg toward 40 deg "
                    "earns progressively more; 45 deg reaches the maximum."
                ),
                "mapping": "18 deg of change or less = 60; 45 deg or more = 240",
                "weight": "25% of Pressure",
            },
        ],
    },
    "balance": {
        "label": "Balance",
        "color": "#f59e0b",
        "score_series": [
            {
                "key": "overall",
                "label": "Balance score",
                "path": "pillars.balance.score",
                "color": "#eef6ff",
                "unit": "/240",
                "default_visible": True,
            },
            {
                "key": "lateral_spread",
                "label": "Lateral spread",
                "path": "pillars.balance.components.lateral_spread.score",
                "color": "#f59e0b",
                "unit": "/240",
                "weight": 35,
                "default_visible": True,
            },
            {
                "key": "smoothness",
                "label": "Movement smoothness",
                "path": "pillars.balance.components.smoothness.score",
                "color": "#6aafff",
                "unit": "/240",
                "weight": 40,
                "default_visible": True,
            },
            {
                "key": "rhythm",
                "label": "Rhythm and symmetry",
                "path": "pillars.balance.components.rhythm.score",
                "color": "#a78bfa",
                "unit": "/240",
                "weight": 25,
                "default_visible": True,
            },
        ],
        "value_series": [
            {
                "key": "lateral_spread",
                "label": "Lateral spread",
                "path": "pillars.balance.components.lateral_spread.value",
                "color": "#f59e0b",
                "unit": "% body height",
                "axis": "percent",
                "default_visible": True,
            },
            {
                "key": "smoothness",
                "label": "Smoothness quality",
                "path": "pillars.balance.components.smoothness.value",
                "color": "#6aafff",
                "unit": "% quality",
                "axis": "percent",
                "default_visible": True,
            },
            {
                "key": "rhythm",
                "label": "Rhythm quality",
                "path": "pillars.balance.components.rhythm.value",
                "color": "#a78bfa",
                "unit": "% quality",
                "axis": "percent",
                "default_visible": True,
            },
            {
                "key": "com_offset",
                "label": "CoM lateral offset",
                "path": "pillars.balance.components.com_offset.value",
                "color": "#fb7185",
                "unit": "% body height",
                "axis": "percent",
                "default_visible": False,
                "contributes_to_score": False,
            },
            {
                "key": "plumb_angle",
                "label": "CoM plumb angle",
                "path": "pillars.balance.components.plumb_angle.value",
                "color": "#f97316",
                "unit": "deg",
                "axis": "degrees",
                "default_visible": False,
                "contributes_to_score": False,
            },
        ],
        "references": [
            {
                "label": "Lateral spread",
                "explanation": (
                    "Over the recent movement window, this measures how widely "
                    "the skier's center of mass travels left and right around "
                    "the midpoint between the boots. A narrow or nearly static "
                    "path scores lower; meaningful movement to both sides "
                    "raises the score."
                ),
                "mapping": "15% body height ~= 140; 18% ~= 150; 20% ~= 221; 25% = 240",
                "weight": "35% of Balance",
            },
            {
                "label": "Movement smoothness",
                "explanation": (
                    "This checks whether the center-of-mass path changes "
                    "gradually. Continuous movement with little sudden "
                    "acceleration or frame-to-frame jitter scores higher; "
                    "abrupt corrections and wobble score lower."
                ),
                "mapping": "0% smoothness quality = 60; 100% = 240",
                "weight": "40% of Balance",
            },
            {
                "label": "Rhythm and symmetry",
                "explanation": (
                    "This compares successive left and right movements. Similar "
                    "movement size, similar timing, and balanced travel on both "
                    "sides earn a higher score; one-sided or irregular movement "
                    "earns less."
                ),
                "mapping": "0% rhythm quality = 60; 100% = 240",
                "weight": "25% of Balance",
            },
            {
                "label": "CoM offset and plumb angle",
                "explanation": (
                    "These lines show where the estimated center of mass sits "
                    "relative to the support point between the boots on each "
                    "frame. They help a coach interpret the movement but do not "
                    "change the Balance score."
                ),
                "mapping": "Visual geometry only; no score is assigned",
                "weight": "Display only",
            },
        ],
    },
    "rotation": {
        "label": "Rotation",
        "color": "#f97316",
        "score_series": [
            {
                "key": "overall",
                "label": "Rotation score",
                "path": "pillars.rotation.score",
                "color": "#eef6ff",
                "unit": "/240",
                "default_visible": True,
            },
            {
                "key": "body_separation",
                "label": "Body separation dynamics",
                "path": "pillars.rotation.components.body_separation.score",
                "color": "#f97316",
                "unit": "/240",
                "weight": 60,
                "default_visible": True,
            },
            {
                "key": "lateral_movement",
                "label": "Lateral movement",
                "path": "pillars.rotation.components.lateral_movement.score",
                "color": "#6aafff",
                "unit": "/240",
                "weight": 30,
                "default_visible": True,
            },
            {
                "key": "direction_change",
                "label": "Direction change",
                "path": "pillars.rotation.components.direction_change.score",
                "color": "#fbbf24",
                "unit": "/240",
                "weight": 10,
                "default_visible": True,
            },
        ],
        "value_series": [
            {
                "key": "body_separation",
                "label": "2D body separation",
                "path": "pillars.rotation.components.body_separation.value",
                "color": "#f97316",
                "unit": "deg",
                "axis": "degrees",
                "default_visible": True,
            },
            {
                "key": "lateral_movement",
                "label": "Lateral movement",
                "path": "pillars.rotation.components.lateral_movement.value",
                "color": "#6aafff",
                "unit": "px/frame",
                "axis": "speed",
                "default_visible": True,
            },
            {
                "key": "direction_change",
                "label": "Direction change rate",
                "path": "pillars.rotation.components.direction_change.value",
                "color": "#fbbf24",
                "unit": "deg/s",
                "axis": "rate",
                "default_visible": False,
            },
        ],
        "references": [
            {
                "label": "Body separation dynamics",
                "explanation": (
                    "This follows the 2D upper-to-lower-body lead angle as the "
                    "skier moves through repeated left and right transitions. "
                    "The angle should expand and reduce smoothly; simply holding "
                    "one large angle does not earn a high score."
                ),
                "mapping": "Less than 5 deg change = 60; about 16 deg with smooth, consistent cycles can reach 240",
                "weight": "60% of Rotation",
            },
            {
                "label": "Lateral movement",
                "explanation": (
                    "This measures horizontal travel using the latest five valid "
                    "frames. More detected side-to-side speed earns a higher "
                    "score, while slow movement below 30 px/frame is penalized "
                    "more strongly."
                ),
                "mapping": "0 px/frame = 60; 30 px/frame = 150; 45 px/frame or more = 240",
                "weight": "30% of Rotation",
            },
            {
                "label": "Direction change",
                "explanation": (
                    "This measures how quickly the detected ski direction changes "
                    "across recent valid ski observations. Faster credible "
                    "changes earn more; implausible jumps are rejected as "
                    "tracking errors."
                ),
                "mapping": "0 deg/s = 60; 6 deg/s or more = 240",
                "weight": "10% of Rotation",
            },
        ],
    },
    "edging": {
        "label": "Edging",
        "color": "#6aafff",
        "score_series": [
            {
                "key": "overall",
                "label": "Edging score",
                "path": "pillars.edging.score",
                "color": "#eef6ff",
                "unit": "/240",
                "default_visible": True,
            },
            {
                "key": "parallelism",
                "label": "Ski parallelism",
                "path": "pillars.edging.components.parallelism.score",
                "color": "#6aafff",
                "unit": "/240",
                "weight": 70,
                "default_visible": True,
            },
            {
                "key": "lateral_movement",
                "label": "Lateral movement",
                "path": "pillars.edging.components.lateral_movement.score",
                "color": "#22c55e",
                "unit": "/240",
                "weight": 30,
                "default_visible": True,
            },
        ],
        "value_series": [
            {
                "key": "parallelism",
                "label": "Angle between skis",
                "path": "pillars.edging.components.parallelism.value",
                "color": "#6aafff",
                "unit": "deg",
                "axis": "degrees",
                "default_visible": True,
            },
            {
                "key": "lateral_movement",
                "label": "Lateral movement",
                "path": "pillars.edging.components.lateral_movement.value",
                "color": "#22c55e",
                "unit": "px/frame",
                "axis": "speed",
                "default_visible": True,
            },
        ],
        "references": [
            {
                "label": "Ski parallelism",
                "explanation": (
                    "This measures the angle between the two detected ski lines. "
                    "Skis pointing in nearly the same direction earn a higher "
                    "score; a wider wedge or larger difference between their "
                    "directions lowers it."
                ),
                "mapping": "1 deg difference or less = 240; 20 deg or more = 60",
                "weight": "70% of Edging",
            },
            {
                "label": "Lateral movement",
                "explanation": (
                    "This uses the latest five valid frames to measure how "
                    "quickly the skier travels across the slope. More detected "
                    "side-to-side speed raises this component; slow movement "
                    "below 30 px/frame receives a stronger reduction."
                ),
                "mapping": "0 px/frame = 60; 30 px/frame = 150; 45 px/frame or more = 240",
                "weight": "30% of Edging",
            },
        ],
    },
}


def sanitize_parameter_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply current coach copy and remove retired graph-only metadata."""
    sanitized = deepcopy(config or TIMELINE_PARAMETER_CONFIG)
    edging = sanitized.get("edging")
    if isinstance(edging, dict):
        for group in ("score_series", "value_series"):
            series = edging.get(group)
            if isinstance(series, list):
                edging[group] = [
                    item
                    for item in series
                    if not isinstance(item, dict)
                    or item.get("key") != "edge_angle"
                ]

        references = edging.get("references")
        if isinstance(references, list):
            edging["references"] = [
                item
                for item in references
                if not isinstance(item, dict)
                or item.get("label") != "2D edge reference"
            ]

    # Timeline rows snapshot their original scoring metadata. Refresh only the
    # explanatory copy so historical runs receive clearer coach guidance while
    # retaining their saved series paths and weights.
    for pillar_key, current_pillar in TIMELINE_PARAMETER_CONFIG.items():
        saved_pillar = sanitized.get(pillar_key)
        if not isinstance(saved_pillar, dict):
            continue
        current_references = {
            reference.get("label"): reference
            for reference in current_pillar.get("references", [])
            if isinstance(reference, dict) and reference.get("label")
        }
        for reference in saved_pillar.get("references", []):
            if not isinstance(reference, dict):
                continue
            current_reference = current_references.get(reference.get("label"))
            if not current_reference:
                continue
            reference["explanation"] = current_reference.get("explanation", "")
            reference["mapping"] = current_reference.get(
                "mapping",
                reference.get("mapping", ""),
            )
    return sanitized


def _number(value: Any, multiplier: float = 1.0) -> Optional[float]:
    try:
        converted = float(value) * multiplier
    except (TypeError, ValueError):
        return None
    if not isfinite(converted):
        return None
    return round(converted, 4)


def _quality_percent(score: Any) -> Optional[float]:
    numeric_score = _number(score)
    if numeric_score is None:
        return None
    return round(max(0.0, min(100.0, (numeric_score - 60.0) / 1.8)), 2)


def _direction_change_rate(score: Any) -> Optional[float]:
    """Invert the active score mapping: BlueIQ = 60 + 30 * deg/s."""
    numeric_score = _number(score)
    if numeric_score is None:
        return None
    return round(max(0.0, min(6.0, (numeric_score - 60.0) / 30.0)), 3)


def build_timeline_sample(
    *,
    time_seconds: Any,
    pillar_scores: Dict[str, Any],
    score_details: Dict[str, Any],
    pressure_knee_angle: Any,
    balance_measurement: Any,
    rotation_separation_angle: Any,
    ski_direction_angle: Any,
    ski_parallelism_angle: Any,
    lateral_speed: Any,
    upper_body_alignment: Any,
    athletic_stance_knee: Any,
    athletic_stance_bend: Any,
) -> Dict[str, Any]:
    """Build one JSON-safe timeline sample from the active scoring path."""
    balance_spread_score = (
        balance_measurement.spread_score if balance_measurement else None
    )
    balance_smoothness_score = (
        balance_measurement.smoothness_score if balance_measurement else None
    )
    balance_rhythm_score = (
        balance_measurement.rhythm_score if balance_measurement else None
    )
    direction_change_score = score_details.get(
        "rotation_direction_change_score"
    )

    sample = {
        "time_seconds": _number(time_seconds),
        "pillars": {
            "pressure": {
                "score": _number(pillar_scores.get("pressure")),
                "components": {
                    "vertical_range": {
                        "score": _number(score_details.get("pressure_vertical_score")),
                        "value": _number(
                            score_details.get("pressure_vertical_range"), 100.0
                        ),
                    },
                    "knee_flexion_range": {
                        "score": _number(
                            score_details.get("pressure_knee_range_score")
                        ),
                        "value": _number(
                            score_details.get("pressure_knee_range")
                        ),
                    },
                    "current_knee_flexion": {
                        "value": _number(pressure_knee_angle),
                    },
                },
            },
            "balance": {
                "score": _number(pillar_scores.get("balance")),
                "components": {
                    "lateral_spread": {
                        "score": _number(balance_spread_score),
                        "value": _number(
                            balance_measurement.spread_value
                            if balance_measurement else None,
                            100.0,
                        ),
                    },
                    "smoothness": {
                        "score": _number(balance_smoothness_score),
                        "value": _quality_percent(balance_smoothness_score),
                    },
                    "rhythm": {
                        "score": _number(balance_rhythm_score),
                        "value": _quality_percent(balance_rhythm_score),
                    },
                    "com_offset": {
                        "value": _number(
                            balance_measurement.normalized_offset
                            if balance_measurement else None,
                            100.0,
                        ),
                    },
                    "plumb_angle": {
                        "value": _number(
                            balance_measurement.offset_angle
                            if balance_measurement else None
                        ),
                    },
                },
            },
            "rotation": {
                "score": _number(pillar_scores.get("rotation")),
                "components": {
                    "body_separation": {
                        "score": _number(
                            score_details.get("rotation_separation_score")
                        ),
                        "value": _number(rotation_separation_angle),
                    },
                    "lateral_movement": {
                        "score": _number(
                            score_details.get("rotation_lateral_score")
                        ),
                        "value": _number(lateral_speed),
                    },
                    "direction_change": {
                        "score": _number(direction_change_score),
                        "value": _direction_change_rate(direction_change_score),
                    },
                    "ski_direction_angle": {
                        "value": _number(ski_direction_angle),
                    },
                },
            },
            "edging": {
                "score": _number(pillar_scores.get("edging")),
                "components": {
                    "parallelism": {
                        "score": _number(
                            score_details.get("edging_parallelism_score")
                        ),
                        "value": _number(ski_parallelism_angle),
                    },
                    "lateral_movement": {
                        "score": _number(
                            score_details.get("edging_lateral_score")
                        ),
                        "value": _number(lateral_speed),
                    },
                },
            },
        },
    }
    # Keep the established report inputs available while the chart consumes
    # the structured pillar/component hierarchy above.
    sample.update({
        "pressure": _number(pillar_scores.get("pressure")),
        "balance": _number(pillar_scores.get("balance")),
        "rotation": _number(pillar_scores.get("rotation")),
        "edging": _number(pillar_scores.get("edging")),
        "ski_parallel_control": _number(ski_parallelism_angle),
        "upper_body_alignment": _number(upper_body_alignment),
        "athletic_stance_knee": _number(athletic_stance_knee),
        "athletic_stance_bend": _number(athletic_stance_bend),
        "transition_control": _number(lateral_speed),
    })
    return sample


def create_timeline_payload(
    samples: Iterable[Dict[str, Any]],
    sample_rate_hz: Any,
) -> Dict[str, Any]:
    return {
        "sample_rate_hz": _number(sample_rate_hz) or 10.0,
        "scoring_version": SCORING_VERSION,
        "samples": list(samples),
        "parameter_config": sanitize_parameter_config(TIMELINE_PARAMETER_CONFIG),
    }


def _flatten_numbers(
    value: Any,
    prefix: str = "",
    output: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    output = output if output is not None else {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numbers(child, child_prefix, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = _number(value)
        if numeric_value is not None:
            output[prefix] = numeric_value
    return output


def _assign_path(target: Dict[str, Any], path: str, value: Any) -> None:
    cursor = target
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def aggregate_timeline_samples(
    samples: Iterable[Dict[str, Any]],
    resolution: str = "second",
) -> List[Dict[str, Any]]:
    """Return frame samples or averages grouped into one-second buckets."""
    valid_samples = [
        sample
        for sample in samples
        if _number(sample.get("time_seconds")) is not None
    ]
    if resolution == "frame":
        return valid_samples
    if resolution != "second":
        raise ValueError("resolution must be 'second' or 'frame'")

    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in valid_samples:
        buckets[int(floor(float(sample["time_seconds"])))].append(sample)

    aggregated: List[Dict[str, Any]] = []
    for second in sorted(buckets):
        values_by_path: Dict[str, List[float]] = defaultdict(list)
        for sample in buckets[second]:
            for path, value in _flatten_numbers(sample).items():
                if path != "time_seconds":
                    values_by_path[path].append(value)

        aggregate: Dict[str, Any] = {
            "time_seconds": second,
            "sample_count": len(buckets[second]),
        }
        for path, values in values_by_path.items():
            _assign_path(aggregate, path, round(sum(values) / len(values), 3))
        aggregated.append(aggregate)

    return aggregated
