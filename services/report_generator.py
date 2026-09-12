import logging
import math
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from services.personal_bests import compare_current_scores_to_previous

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

logger = logging.getLogger(__name__)

MAX_SCORE = 240
PAGE_W = 1275
PAGE_H = 1650

SCORE_BANDS = (
    ("Building", 60, 104, (249, 115, 22)),      # Orange
    ("Progressing", 105, 149, (56, 189, 248)),  # Sky blue
    ("Advancing", 150, 194, (106, 175, 255)),   # Blue
    ("Mastering", 195, 240, (34, 197, 94)),     # Green
)

FACTOR_KEYS = (
    "ski_parallel_control",
    "edge_control",
    "upper_body_alignment",
    "athletic_stance_knee",
    "athletic_stance_bend",
    "transition_control",
)

COACHING_FACTOR_LABELS = {
    "ski_parallel_control": "Ski parallel control",
    "edge_control": "Edge control",
    "upper_body_alignment": "Upper-body alignment",
    "athletic_stance_knee": "Knee flexion",
    "athletic_stance_bend": "Athletic stance",
    "transition_control": "Turn-to-turn transition control",
}

PILLAR_GUIDANCE = {
    "pressure": "Pressure reflects how consistently the skier manages load and release through each turn.",
    "balance": "Balance reflects centered body position, stability, and control through the fall line.",
    "rotation": "Rotation reflects upper-body discipline and how cleanly the skier directs each turn.",
    "edging": "Edging reflects edge engagement, ski parallel control, and ability to hold a clean arc.",
}

# 12 Sub-levels: 4 stages x 3 sub-levels (15 points per step across 60-240)
BLUEIQ_LEVEL_GUIDE = (
    {
        "level": 1,
        "stage": "Building",
        "sub_level": 1,
        "name": "Building 1",
        "category": "Building",
        "low": 60,
        "high": 74,
        "status": "Establishing baseline comfort and movement on skis.",
        "key_focus": "Equipment familiarity, centered stance, basic gliding, wedge turns, and safe stopping.",
    },
    {
        "level": 2,
        "stage": "Building",
        "sub_level": 2,
        "name": "Building 2",
        "category": "Building",
        "low": 75,
        "high": 89,
        "status": "Building confidence and balance through turns.",
        "key_focus": "Linking turns smoothly, developing foot tilt, and improving speed control.",
    },
    {
        "level": 3,
        "stage": "Building",
        "sub_level": 3,
        "name": "Building 3",
        "category": "Building",
        "low": 90,
        "high": 104,
        "status": "Demonstrating steady progression and consistent turn control.",
        "key_focus": "Refining wedge christie turns, upper body stability, and cleaner transitions.",
    },
    {
        "level": 4,
        "stage": "Progressing",
        "sub_level": 1,
        "name": "Progressing 1",
        "category": "Progressing",
        "low": 105,
        "high": 119,
        "status": "Beginning parallel ski matching with growing rhythm.",
        "key_focus": "Parallel matching before turn completion, edge engagement, and maintaining centered balance.",
    },
    {
        "level": 5,
        "stage": "Progressing",
        "sub_level": 2,
        "name": "Progressing 2",
        "category": "Progressing",
        "low": 120,
        "high": 134,
        "status": "Skiing with smoother flow, fluidity, and coordinated edge control.",
        "key_focus": "Active knee flexion, side slipping, fluid turn transitions, and speed management.",
    },
    {
        "level": 6,
        "stage": "Progressing",
        "sub_level": 3,
        "name": "Progressing 3",
        "category": "Progressing",
        "low": 135,
        "high": 149,
        "status": "Demonstrating solid parallel turns with consistent tempo.",
        "key_focus": "Beginning turns with parallel skis, refined pressure modulation, and disciplined upper body.",
    },
    {
        "level": 7,
        "stage": "Advancing",
        "sub_level": 1,
        "name": "Advancing 1",
        "category": "Advancing",
        "low": 150,
        "high": 164,
        "status": "Advancing technical precision across turn shapes.",
        "key_focus": "Consistent parallel steering, progressive edge angulation, and centered dynamic stance.",
    },
    {
        "level": 8,
        "stage": "Advancing",
        "sub_level": 2,
        "name": "Advancing 2",
        "category": "Advancing",
        "low": 165,
        "high": 179,
        "status": "Executing high-level dynamic carving and precise pressure release.",
        "key_focus": "Early edge engagement, upper-lower body separation, and rhythmic speed control.",
    },
    {
        "level": 9,
        "stage": "Advancing",
        "sub_level": 3,
        "name": "Advancing 3",
        "category": "Advancing",
        "low": 180,
        "high": 194,
        "status": "Approaching master-level consistency and power through the turn arc.",
        "key_focus": "Dynamic lateral angulation, precise edge transition, and disciplined fall-line control.",
    },
    {
        "level": 10,
        "stage": "Mastering",
        "sub_level": 1,
        "name": "Mastering 1",
        "category": "Mastering",
        "low": 195,
        "high": 209,
        "status": "Mastering technical precision, speed, and continuous arc control.",
        "key_focus": "High-velocity carving, rapid edge changes, and full biomechanical efficiency.",
    },
    {
        "level": 11,
        "stage": "Mastering",
        "sub_level": 2,
        "name": "Mastering 2",
        "category": "Mastering",
        "low": 210,
        "high": 224,
        "status": "Demonstrating elite ski performance and exceptional dynamic balance.",
        "key_focus": "Flawless pressure management, advanced carving versatility, and deep athletic angulation.",
    },
    {
        "level": 12,
        "stage": "Mastering",
        "sub_level": 3,
        "name": "Mastering 3",
        "category": "Mastering",
        "low": 225,
        "high": 240,
        "status": "Peak technical mastery across all biomechanical performance pillars.",
        "key_focus": "Complete mastery of edging, rotation, pressure, and balance in all conditions.",
    },
)

APPROVED_COACHING_CUES = {
    "pressure": {
        "building": "Build a smoother pressure release so the skier is not forced back at the end of the turn.",
        "progressing": "Work on applying pressure progressively across both skis instead of loading the turn all at once.",
        "advancing": "Refine pressure timing so the skier can stay strong through the middle of the turn.",
        "mastering": "Maintain the same pressure control while increasing turn shape and speed.",
        "emerging": "Build a smoother pressure release so the skier is not forced back at the end of the turn.",
        "developing": "Work on applying pressure progressively across both skis instead of loading the turn all at once.",
        "proficient": "Refine pressure timing so the skier can stay strong through the middle of the turn.",
        "excellent": "Maintain the same pressure control while increasing turn shape and speed.",
    },
    "balance": {
        "building": "Focus on staying centered over the feet before adding more speed or turn shape.",
        "progressing": "Build a quieter upper body and keep the stance centered through each transition.",
        "advancing": "Refine balance consistency as the skier moves from one edge to the next.",
        "mastering": "Keep the same centered stance while challenging the skier with more varied turn shapes.",
        "emerging": "Focus on staying centered over the feet before adding more speed or turn shape.",
        "developing": "Build a quieter upper body and keep the stance centered through each transition.",
        "proficient": "Refine balance consistency as the skier moves from one edge to the next.",
        "excellent": "Keep the same centered stance while challenging the skier with more varied turn shapes.",
    },
    "rotation": {
        "building": "Keep the upper body calmer so the skis can finish the turn without extra twisting.",
        "progressing": "Work on separating upper-body discipline from lower-body turning action.",
        "advancing": "Refine rotational control so the skier directs the turn without over-rotating the shoulders.",
        "mastering": "Maintain disciplined rotation while increasing tempo and precision.",
        "emerging": "Keep the upper body calmer so the skis can finish the turn without extra twisting.",
        "developing": "Work on separating upper-body discipline from lower-body turning action.",
        "proficient": "Refine rotational control so the skier directs the turn without over-rotating the shoulders.",
        "excellent": "Maintain disciplined rotation while increasing tempo and precision.",
    },
    "edging": {
        "building": "Start with cleaner ski alignment so both skis work together earlier in the turn.",
        "progressing": "Improve edge engagement by keeping the skis more parallel through the turn entry.",
        "advancing": "Refine edge control so the skier can hold a cleaner arc through the fall line.",
        "mastering": "Preserve strong edge control while increasing speed and consistency.",
        "emerging": "Start with cleaner ski alignment so both skis work together earlier in the turn.",
        "developing": "Improve edge engagement by keeping the skis more parallel through the turn entry.",
        "proficient": "Refine edge control so the skier can hold a cleaner arc through the fall line.",
        "excellent": "Preserve strong edge control while increasing speed and consistency.",
    },
}

APPROVED_PILLAR_ASSESSMENTS = {
    "pressure": {
        "building": "Pressure management is building, with developing consistency in how load is built and released through the turn.",
        "progressing": "Pressure management is progressing, with some controlled loading and release but developing timing across the run.",
        "advancing": "Pressure control is advancing, with generally consistent loading and release through most turns.",
        "mastering": "Pressure control is mastering, with precise and consistent loading and release across the run.",
        "emerging": "Pressure management is emerging, with limited consistency in how load is built and released through the turn.",
        "developing": "Pressure management is developing, with some controlled loading and release but inconsistent timing across the run.",
        "proficient": "Pressure control is proficient, with generally consistent loading and release through most turns.",
        "excellent": "Pressure control is excellent, with precise and consistent loading and release across the run.",
    },
    "balance": {
        "building": "Balance is building, with centered stance and stability developing through the run.",
        "progressing": "Balance is progressing, with periods of centered control and growing stability through transitions.",
        "advancing": "Balance is advancing, with a generally centered stance and stable movement through most transitions.",
        "mastering": "Balance is mastering, with an athletic centered stance and consistent stability throughout the run.",
        "emerging": "Balance is emerging, with centered stance and stability remaining inconsistent through the run.",
        "developing": "Balance is developing, with periods of centered control but uneven stability through transitions.",
        "proficient": "Balance is proficient, with a generally centered stance and stable movement through most transitions.",
        "excellent": "Balance is excellent, with a centered stance and consistent stability throughout the run.",
    },
    "rotation": {
        "building": "Rotational control is building, with upper- and lower-body coordination developing.",
        "progressing": "Rotational control is progressing, with effective separation developing alongside upper-body discipline.",
        "advancing": "Rotational control is advancing, with coordinated lower-body turning and generally disciplined upper-body movement.",
        "mastering": "Rotational control is mastering, with precise upper- and lower-body coordination throughout the run.",
        "emerging": "Rotational control is emerging, with upper- and lower-body coordination remaining inconsistent.",
        "developing": "Rotational control is developing, with some effective separation but inconsistent upper-body discipline.",
        "proficient": "Rotational control is proficient, with coordinated lower-body turning and generally disciplined upper-body movement.",
        "excellent": "Rotational control is excellent, with precise upper- and lower-body coordination throughout the run.",
    },
    "edging": {
        "building": "Edging is building, with ski alignment and edge engagement developing.",
        "progressing": "Edging is progressing, with periods of parallel ski control and developing engagement through the turn.",
        "advancing": "Edging is advancing, with generally parallel ski control and consistent engagement through most turns.",
        "mastering": "Edging is mastering, with precise ski alignment and consistent engagement throughout the run.",
        "emerging": "Edging is emerging, with ski alignment and edge engagement remaining inconsistent.",
        "developing": "Edging is developing, with periods of parallel ski control but inconsistent engagement through the turn.",
        "proficient": "Edging is proficient, with generally parallel ski control and consistent engagement through most turns.",
        "excellent": "Edging is excellent, with precise ski alignment and consistent engagement throughout the run.",
    },
}

APPROVED_OVERALL_ASSESSMENTS = {
    "building": "The result reflects a building foundation, with consistency and control requiring focused development.",
    "progressing": "The result reflects progressing control, with repeatable movement alongside clear opportunities for greater consistency.",
    "advancing": "The result reflects advancing control across the run, with targeted refinement needed for greater consistency.",
    "mastering": "The result reflects mastering control and consistency across the four performance pillars.",
    "emerging": "The result reflects an emerging foundation, with consistency and control requiring focused development.",
    "developing": "The result reflects developing control, with repeatable movement alongside clear opportunities for greater consistency.",
    "proficient": "The result reflects proficient control across the run, with targeted refinement needed for greater consistency.",
    "excellent": "The result reflects excellent control and consistency across the four performance pillars.",
}

BASELINE_COMPARISON_PATTERN = re.compile(
    r"\b(?:improved|improving|progressed|recovered)\b"
    r"|\b(?:increased|decreased)\s+by\b"
    r"|\bcompared\s+(?:with|to)\b",
    re.IGNORECASE,
)


def _score_level_label(score: float) -> str:
    rounded = _ceil_score(score)
    for guide in BLUEIQ_LEVEL_GUIDE:
        if guide["low"] <= rounded <= guide["high"]:
            return guide["name"]
    return BLUEIQ_LEVEL_GUIDE[0]["name"] if rounded < 60 else BLUEIQ_LEVEL_GUIDE[-1]["name"]


def _score_stage(score: float) -> str:
    rounded = _ceil_score(score)
    for label, low, high, _color in SCORE_BANDS:
        if low <= rounded <= high:
            return label
    return SCORE_BANDS[0][0] if rounded < 60 else SCORE_BANDS[-1][0]


def _score_band(score: float) -> str:
    """Return the 12-step stage and sub-level label (e.g. 'Building 2', 'Progressing 1')."""
    return _score_level_label(score)


def _score_color(score: float) -> tuple:
    rounded = _ceil_score(score)
    for _label, low, high, color in SCORE_BANDS:
        if low <= rounded <= high:
            return color
    return SCORE_BANDS[0][3] if rounded < 60 else SCORE_BANDS[-1][3]


def _score_band_key(score: float) -> str:
    return _score_stage(score).lower()


def _approved_cue(pillar: str, score: float) -> str:
    pillar_cues = APPROVED_COACHING_CUES.get(pillar, {})
    return pillar_cues.get(_score_band_key(score)) or PILLAR_GUIDANCE.get(pillar, "")


def _approved_assessment(pillar: str, score: float) -> str:
    pillar_assessments = APPROVED_PILLAR_ASSESSMENTS.get(pillar, {})
    return (
        pillar_assessments.get(_score_band_key(score))
        or PILLAR_GUIDANCE.get(pillar, "")
    )


def _approved_improvement_areas(scores: Dict[str, Any], limit: int = 3) -> List[str]:
    ranked = sorted(
        (
            ("pressure", scores["pressure"]),
            ("balance", scores["balance"]),
            ("rotation", scores["rotation"]),
            ("edging", scores["edging"]),
        ),
        key=lambda item: item[1],
    )
    return [_approved_cue(pillar, score) for pillar, score in ranked[:limit]]


def _blueiq_level_guide(score: float) -> Dict[str, Any]:
    rounded = _ceil_score(score)
    for guide in BLUEIQ_LEVEL_GUIDE:
        if guide["low"] <= rounded <= guide["high"]:
            return dict(guide)
    if rounded < BLUEIQ_LEVEL_GUIDE[0]["low"]:
        return dict(BLUEIQ_LEVEL_GUIDE[0])
    return dict(BLUEIQ_LEVEL_GUIDE[-1])


def _blueiq_score_interpretation(score: float, guide: Dict[str, Any]) -> str:
    rounded = _ceil_score(score)
    for low, high, text in guide.get("interpretations", ()):
        if low <= rounded <= high:
            return text
    return guide.get("status", "")


def _report_level_context(score: float) -> Dict[str, Any]:
    guide = _blueiq_level_guide(score)
    return {
        "category": guide["category"],
        "level": guide["level"],
        "name": guide["name"],
        "score_range": f"{guide['low']}-{guide['high']}",
        "status": guide["status"],
        "key_focus": guide["key_focus"],
        "score_interpretation": _blueiq_score_interpretation(score, guide),
    }


def _level_context_sentence(payload: Dict[str, Any]) -> str:
    level = payload.get("blueiq_level") or _report_level_context(payload["final_scores"]["blue_iq"])
    return f"Blue IQ places this skier at {level['name']}."


def _level_focus_cue(payload: Dict[str, Any]) -> str:
    level = payload.get("blueiq_level") or _report_level_context(payload["final_scores"]["blue_iq"])
    return f"Current stage focus ({level['name']}): {level['key_focus']}"


def _safe_average(values: List[float]) -> Optional[float]:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 1)


def _round_score(value: Any) -> Optional[int]:
    if not isinstance(value, (int, float)):
        return None
    return int(round(float(value)))


def _ceil_score(value: Any) -> int:
    return int(math.ceil(float(value or 0)))


def build_score_windows(score_timeline: List[Dict[str, Any]], window_seconds: int = 10) -> List[Dict[str, Any]]:
    """Average pillar scores and coaching factors into fixed time windows."""
    windows: Dict[int, Dict[str, Any]] = {}
    pillar_keys = ("pressure", "balance", "rotation", "edging")

    for item in score_timeline:
        time_sec = float(item.get("time_seconds") or 0)
        window_index = int(time_sec // window_seconds)
        bucket = windows.setdefault(
            window_index,
            {
                "start_seconds": window_index * window_seconds,
                "end_seconds": (window_index + 1) * window_seconds,
                **{key: [] for key in pillar_keys},
                **{key: [] for key in FACTOR_KEYS},
            },
        )
        for key in (*pillar_keys, *FACTOR_KEYS):
            value = item.get(key)
            if isinstance(value, (int, float)):
                bucket[key].append(float(value))

    averaged = []
    for index in sorted(windows):
        bucket = windows[index]
        row = {
            "window": f"{bucket['start_seconds']}-{bucket['end_seconds']} sec",
            "pillar_scores": {},
            "coaching_factors": {},
        }
        for key in pillar_keys:
            row["pillar_scores"][key] = _safe_average(bucket[key])
        for key in FACTOR_KEYS:
            row["coaching_factors"][key] = _safe_average(bucket[key])
        averaged.append(row)

    return averaged


def _load_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None

    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_session_font(size: int):
    if ImageFont is None:
        return None
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "arialbd.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return _load_font(size, bold=True)


def _format_session_date(session_date):
    if not session_date:
        return None
    if isinstance(session_date, datetime):
        value = session_date.date()
    elif isinstance(session_date, date):
        value = session_date
    else:
        text = str(session_date).strip()
        try:
            value = datetime.fromisoformat(text).date()
        except ValueError:
            return text
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _text_width(draw, text: str, font) -> int:
    return int(draw.textbbox((0, 0), text, font=font)[2])


def _wrap_text(draw, text: str, font, max_width: int) -> List[str]:
    words = str(text or "").replace("\n", " ").split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_wrapped(draw, text: str, x: int, y: int, max_width: int, font, fill, line_gap: int = 8, max_lines: Optional[int] = None) -> int:
    lines = _wrap_text(draw, text, font, max_width)
    if max_lines:
        lines = lines[:max_lines]
    line_h = int(font.size * 1.25) if hasattr(font, "size") else 22
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h + line_gap
    return y


def _fit_text_to_lines(draw, text: str, font, max_width: int, max_lines: int) -> str:
    """Keep text inside a card without chopping mid-sentence."""
    text = _clean_report_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = " ".join([*kept, sentence]).strip()
        if len(_wrap_text(draw, candidate, font, max_width)) <= max_lines:
            kept.append(sentence)
        else:
            break
    if kept:
        return " ".join(kept)

    lines = _wrap_text(draw, text, font, max_width)
    clipped = " ".join(lines[:max_lines]).strip()
    if clipped and clipped[-1] not in ".!?":
        clipped = clipped.rstrip(",;:") + "."
    return clipped


def _draw_badge(draw, text: str, x: int, y: int, color: tuple, font) -> None:
    text_w = _text_width(draw, text, font)
    draw.rounded_rectangle((x, y, x + text_w + 34, y + 34), radius=17, fill=(10, 24, 42), outline=color, width=2)
    draw.text((x + 17, y + 7), text, fill=color, font=font)


def _personal_best_context_text(comparison: Optional[Dict[str, Any]]) -> str:
    if not comparison:
        return ""
    status = comparison.get("status")
    record = comparison.get("personal_best") or {}
    score = int(comparison.get("personal_best_score") or 0)
    session_number = int(record.get("session_number") or 0)
    run_number = int(record.get("run_number") or 0)
    record_date = _format_session_date(record.get("date")) or ""
    context_parts = []
    if session_number:
        context_parts.append(f"Session {session_number}")
    if run_number:
        context_parts.append(f"Run {run_number}")
    if record_date:
        context_parts.append(record_date)
    context = " | ".join(context_parts)

    if status == "new_personal_best":
        previous = comparison.get("previous_best") or {}
        previous_score = previous.get("score")
        change = (
            int(comparison["current_score"]) - int(previous_score)
            if previous_score is not None
            else 0
        )
        return f"New personal best +{change}" if change else "New personal best"
    if status == "baseline":
        return f"Baseline established | {context}" if context else "Baseline established"
    if status == "matches_personal_best":
        return f"Matches personal best {score} | {context}".rstrip(" |")
    points_below = int(comparison.get("points_below") or 0)
    return f"Personal best {score} | {points_below} points below | {context}".rstrip(" |")


def _compact_personal_best_text(comparison: Optional[Dict[str, Any]]) -> str:
    if not comparison:
        return ""
    status = comparison.get("status")
    if status == "new_personal_best":
        previous = comparison.get("previous_best") or {}
        previous_score = previous.get("score")
        change = (
            int(comparison["current_score"]) - int(previous_score)
            if previous_score is not None
            else 0
        )
        return f"NEW PB +{change}" if change else "BASELINE"
    if status == "baseline":
        return "BASELINE"
    if status == "matches_personal_best":
        return "MATCHES PB"
    return f"PB {comparison['personal_best_score']} | {comparison['points_below']} BELOW"


def _draw_score_bar(draw, x: int, y: int, w: int, score: float, color: tuple) -> None:
    score = max(0, min(MAX_SCORE, float(score or 0)))
    fill_w = int(w * (score / MAX_SCORE))
    draw.rounded_rectangle((x, y, x + w, y + 10), radius=5, fill=(35, 52, 76))
    draw.rounded_rectangle((x, y, x + fill_w, y + 10), radius=5, fill=color)


def _coerce_text(value: Any, fallback: str = "") -> str:
    """Convert model JSON values into clean report text."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip() or fallback
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return " ".join(part for part in parts if part).strip() or fallback
    if isinstance(value, dict):
        preferred_keys = ("summary", "assessment", "coaching_focus", "focus", "recommendation", "text")
        parts = [_coerce_text(value.get(key)) for key in preferred_keys if key in value]
        if not parts:
            parts = [_coerce_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip() or fallback
    return str(value).strip() or fallback


def _clean_report_text(text: str) -> str:
    text = _coerce_text(text)
    text = re.sub(r"\b(\d+)\.0\b", r"\1", text)
    text = re.sub(r"\b(\d+)\.\d+\b", r"\1", text)
    replacements = {
        "window 1": "the first part of the run",
        "window 2": "the middle part of the run",
        "window 3": "the final part of the run",
        "Window 1": "The first part of the run",
        "Window 2": "The middle part of the run",
        "Window 3": "The final part of the run",
        "10-second window": "part of the run",
        "10 second window": "part of the run",
        "windows": "parts of the run",
        "Windows": "Parts of the run",
        "segments": "parts of the run",
        "segment": "part of the run",
        "Segments": "Parts of the run",
        "Segment": "Part of the run",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _clean_sections(sections: Dict[str, Any]) -> Dict[str, Any]:
    sections["overall"] = _clean_report_text(sections.get("overall"))
    for key, section in sections.get("pillars", {}).items():
        section["summary"] = _clean_report_text(section.get("summary"))
        section["coaching_focus"] = _clean_report_text(section.get("coaching_focus"))
    sections["improvement_areas"] = [
        _clean_report_text(item) for item in sections.get("improvement_areas", []) if _clean_report_text(item)
    ]
    return sections


def _report_comparison_status(payload: Dict[str, Any]) -> str:
    comparison = payload.get("personal_best_comparisons", {}).get("blue_iq", {})
    return str(comparison.get("status") or "baseline")


def _fixed_overall_summary(payload: Dict[str, Any]) -> str:
    scores = payload["final_scores"]
    blue_iq = scores["blue_iq"]
    band_key = _score_band_key(blue_iq)
    band_label = _score_band(blue_iq)
    status = _report_comparison_status(payload)

    if status == "baseline":
        score_context = (
            f"This first recorded run establishes a Blue IQ baseline of "
            f"{blue_iq:.0f}/240 at the {band_label} stage."
        )
    elif status == "new_personal_best":
        score_context = (
            f"This run sets a new personal-best Blue IQ of {blue_iq:.0f}/240 "
            f"at the {band_label} stage."
        )
    else:
        score_context = (
            f"This run records a Blue IQ of {blue_iq:.0f}/240 at the "
            f"{band_label} stage."
        )

    ranked = sorted(
        (
            ("pressure", scores["pressure"]),
            ("balance", scores["balance"]),
            ("rotation", scores["rotation"]),
            ("edging", scores["edging"]),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    strongest_key, strongest_score = ranked[0]
    weakest_key, weakest_score = ranked[-1]
    if strongest_score == weakest_score:
        pillar_context = (
            f"All four pillars recorded {strongest_score:.0f}/240 in this run."
        )
    else:
        pillar_context = (
            f"{strongest_key.title()} is the highest pillar at "
            f"{strongest_score:.0f}/240, while {weakest_key.title()} at "
            f"{weakest_score:.0f}/240 is the main development priority."
        )

    return " ".join((
        score_context,
        _level_context_sentence(payload),
        APPROVED_OVERALL_ASSESSMENTS[band_key],
        pillar_context,
    ))


def _validate_fixed_sections(sections: Dict[str, Any], payload: Dict[str, Any]) -> None:
    scores = payload["final_scores"]
    for pillar in ("pressure", "balance", "rotation", "edging"):
        expected = _approved_assessment(pillar, scores[pillar])
        actual = sections["pillars"][pillar]["summary"]
        if actual != expected:
            raise ValueError(f"{pillar.title()} narrative does not match its score band")

    if _report_comparison_status(payload) == "baseline":
        narrative = " ".join([
            sections["overall"],
            *(
                sections["pillars"][pillar]["summary"]
                for pillar in ("pressure", "balance", "rotation", "edging")
            ),
        ])
        if BASELINE_COMPARISON_PATTERN.search(narrative):
            raise ValueError("Baseline report contains unsupported progress language")


def _build_fixed_sections(payload: Dict[str, Any]) -> Dict[str, Any]:
    scores = payload["final_scores"]
    sections = {
        "overall": _fixed_overall_summary(payload),
        "pillars": {
            pillar: {
                "summary": _approved_assessment(pillar, scores[pillar]),
                "coaching_focus": _approved_cue(pillar, scores[pillar]),
            }
            for pillar in ("pressure", "balance", "rotation", "edging")
        },
        "improvement_areas": [
            _level_focus_cue(payload),
            *_approved_improvement_areas(scores),
        ][:4],
    }
    sections = _clean_sections(sections)
    _validate_fixed_sections(sections, payload)
    return sections


def _run_segment_label(index: int, total: int) -> str:
    if total <= 1:
        return "Full run"
    if index == 0:
        return "First part of the run"
    if index == total - 1:
        return "Final part of the run"
    return "Middle part of the run" if total == 3 else f"Middle part {index}"


def _build_prompt_segments(score_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments = []
    total = len(score_windows)
    for index, row in enumerate(score_windows):
        segment = {
            "part": _run_segment_label(index, total),
            "pillar_scores": {},
            "coaching_factors": {},
        }
        for key, value in row.get("pillar_scores", {}).items():
            segment["pillar_scores"][key] = _ceil_score(value) if isinstance(value, (int, float)) else None
        for key, value in row.get("coaching_factors", {}).items():
            segment["coaching_factors"][key] = _round_score(value)
        segments.append(segment)
    return segments


def _draw_header(canvas: Image.Image, draw, fonts: Dict[str, Any], logo_path: str, payload: Dict[str, Any]) -> None:
    colors = {
        "text": (238, 246, 255),
        "muted": (150, 169, 194),
        "blue": (106, 175, 255),
    }

    logo_loaded = False
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo.thumbnail((270, 82), Image.LANCZOS)
            canvas.paste(logo, (62, 58), logo)
            logo_loaded = True
        except Exception as exc:
            logger.error(f"Unable to load report logo: {exc}")

    if not logo_loaded:
        draw.text((72, 70), "bluerun", fill=colors["blue"], font=fonts["brand"])

    draw.text((72, 142), "Ski Performance Report", fill=colors["text"], font=fonts["title"])
    session_name = str(payload.get("user_name") or "").strip()
    if session_name:
        name_font = _load_session_font(34)
        meta_font = _load_session_font(15)
        name_x, name_y = 72, 196
        draw.text((name_x + 2, name_y + 2), session_name, fill=(0, 0, 0), font=name_font)
        draw.text((name_x, name_y), session_name, fill=(255, 255, 255), font=name_font)
        name_bbox = draw.textbbox((name_x, name_y), session_name, font=name_font)
        line_y = name_bbox[3] + 8
        draw.rectangle((name_x, line_y, name_x + 44, line_y + 4), fill=colors["blue"])

        cursor_x = name_x
        meta_y = line_y + 16
        if payload.get("session_number"):
            session_text = f"SESSION {payload['session_number']}"
            draw.text((cursor_x, meta_y), session_text, fill=colors["blue"], font=meta_font)
            cursor_x += int(draw.textlength(session_text, font=meta_font)) + 14

        if payload.get("session_number") and payload.get("attempt_number"):
            dot_cx = cursor_x + 2
            dot_cy = meta_y + 10
            draw.ellipse((dot_cx - 2, dot_cy - 2, dot_cx + 2, dot_cy + 2), fill=(100, 110, 120))
            cursor_x += 14

        if payload.get("attempt_number"):
            run_text = f"RUN {payload['attempt_number']}"
            draw.text((cursor_x, meta_y), run_text, fill=colors["blue"], font=meta_font)
            cursor_x += int(draw.textlength(run_text, font=meta_font)) + 14

        formatted_date = _format_session_date(payload.get("session_date"))
        if payload.get("attempt_number") and formatted_date:
            dot_cx = cursor_x + 2
            dot_cy = meta_y + 10
            draw.ellipse((dot_cx - 2, dot_cy - 2, dot_cx + 2, dot_cy + 2), fill=(100, 110, 120))
            cursor_x += 14
        if formatted_date:
            draw.text((cursor_x, meta_y), formatted_date, fill=(160, 175, 190), font=meta_font)
    else:
        draw.text((72, 204), "One-page coaching summary generated from Blue IQ analysis", fill=colors["muted"], font=fonts["body"])

    scores = payload["final_scores"]
    blue_iq = scores["blue_iq"]
    color = _score_color(blue_iq)
    draw.rounded_rectangle((865, 58, 1215, 238), radius=28, fill=(10, 24, 42), outline=(34, 69, 110), width=2)
    draw.text((905, 88), "BLUE IQ", fill=colors["blue"], font=fonts["small_bold"])
    draw.text((905, 122), f"{blue_iq:.0f}", fill=colors["text"], font=fonts["score"])
    draw.text((1032, 160), f"/ {MAX_SCORE}", fill=colors["muted"], font=fonts["body_bold"])
    _draw_badge(draw, _score_band(blue_iq).upper(), 905, 190, color, fonts["tiny_bold"])
    comparison = payload.get("personal_best_comparisons", {}).get("blue_iq")
    if comparison and comparison.get("is_new_personal_best"):
        _draw_badge(draw, "NEW PB", 1055, 84, (34, 197, 94), fonts["tiny_bold"])
    compact_text = _compact_personal_best_text(comparison)
    if compact_text:
        draw.text((1040, 202), compact_text, fill=(165, 181, 204), font=fonts["micro_bold"])


def _draw_pillar_card(
    draw,
    x: int,
    y: int,
    w: int,
    h: int,
    name: str,
    score: float,
    section: Dict[str, str],
    fonts: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None,
) -> None:
    color = _score_color(score)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=22, fill=(12, 27, 47), outline=(34, 54, 82), width=2)
    draw.rectangle((x, y + 24, x + 5, y + h - 24), fill=color)
    draw.text((x + 28, y + 26), name, fill=(238, 246, 255), font=fonts["card_title"])
    if comparison and comparison.get("is_new_personal_best"):
        _draw_badge(draw, "NEW PB", x + 155, y + 24, (34, 197, 94), fonts["tiny_bold"])
    draw.text((x + w - 178, y + 25), f"{score:.0f}", fill=(238, 246, 255), font=fonts["metric_score"])
    draw.text((x + w - 88, y + 40), f"/ {MAX_SCORE}", fill=(165, 181, 204), font=fonts["small_bold"])
    _draw_badge(draw, _score_band(score).upper(), x + w - 184, y + 74, color, fonts["tiny_bold"])
    _draw_score_bar(draw, x + 28, y + 90, w - 220, score, color)

    comparison_text = _personal_best_context_text(comparison)
    body_y = y + 122
    max_lines = 4
    if comparison_text:
        fitted_comparison = _fit_text_to_lines(
            draw,
            comparison_text,
            fonts["micro_bold"],
            w - 56,
            1,
        )
        draw.text(
            (x + 28, y + 108),
            fitted_comparison,
            fill=(106, 175, 255),
            font=fonts["micro_bold"],
        )
        body_y = y + 137

    summary = _coerce_text(section.get("summary"))
    coaching_focus = _coerce_text(section.get("coaching_focus"))
    body = _fit_text_to_lines(draw, f"{summary} {coaching_focus}".strip(), fonts["small"], w - 56, max_lines)
    _draw_wrapped(draw, body, x + 28, body_y, w - 56, fonts["small"], (180, 196, 218), line_gap=5, max_lines=max_lines)


def _draw_snapshot_card(canvas: Image.Image, draw, x: int, y: int, w: int, h: int, snapshot_path: str, fonts: Dict[str, Any]) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=28, fill=(10, 24, 42), outline=(34, 54, 82), width=2)
    draw.text((x + 28, y + 26), "Technique Snapshot", fill=(238, 246, 255), font=fonts["section"])

    if not snapshot_path or not os.path.exists(snapshot_path):
        draw.text((x + 28, y + 88), "Snapshot unavailable", fill=(150, 169, 194), font=fonts["body"])
        return

    try:
        image = Image.open(snapshot_path).convert("RGB")
        image.thumbnail((w - 56, h - 98), Image.LANCZOS)
        image_x = x + (w - image.width) // 2
        image_y = y + 76 + max(0, (h - 98 - image.height) // 2)
        draw.rounded_rectangle(
            (image_x - 3, image_y - 3, image_x + image.width + 3, image_y + image.height + 3),
            radius=16,
            fill=(4, 12, 24),
            outline=(34, 69, 110),
            width=2,
        )
        canvas.paste(image, (image_x, image_y))
    except Exception as exc:
        logger.error(f"Unable to draw report snapshot: {exc}")
        draw.text((x + 28, y + 88), "Snapshot unavailable", fill=(150, 169, 194), font=fonts["body"])


def _draw_pdf_report(report_path: str, payload: Dict[str, Any], sections: Dict[str, Any]) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to generate PDF reports")

    fonts = {
        "brand": _load_font(44, bold=True),
        "title": _load_font(40, bold=True),
        "section": _load_font(25, bold=True),
        "card_title": _load_font(25, bold=True),
        "body": _load_font(22),
        "body_bold": _load_font(22, bold=True),
        "small": _load_font(18),
        "small_bold": _load_font(17, bold=True),
        "micro_bold": _load_font(14, bold=True),
        "tiny_bold": _load_font(13, bold=True),
        "score": _load_font(62, bold=True),
        "metric_score": _load_font(34, bold=True),
    }

    canvas = Image.new("RGB", (PAGE_W, PAGE_H), (5, 10, 22))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((-220, -210, 620, 640), fill=(8, 35, 72))
    draw.ellipse((760, -180, 1490, 520), fill=(7, 24, 52))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "bluerun.png")
    _draw_header(canvas, draw, fonts, logo_path, payload)

    snapshot_path = payload.get("snapshot_path")
    has_snapshot = bool(snapshot_path and os.path.exists(snapshot_path))
    if has_snapshot:
        draw.rounded_rectangle((60, 300, 715, 600), radius=28, fill=(10, 24, 42), outline=(34, 54, 82), width=2)
        draw.text((92, 332), "Overall Performance", fill=(238, 246, 255), font=fonts["section"])
        overall_text = _fit_text_to_lines(draw, sections["overall"], fonts["body"], 595, 6)
        _draw_wrapped(draw, overall_text, 92, 380, 595, fonts["body"], (186, 202, 224), line_gap=8, max_lines=6)
        _draw_snapshot_card(canvas, draw, 745, 300, 470, 300, snapshot_path, fonts)
        cards_start_y = 640
    else:
        draw.rounded_rectangle((60, 300, 1215, 500), radius=28, fill=(10, 24, 42), outline=(34, 54, 82), width=2)
        draw.text((92, 332), "Overall Performance", fill=(238, 246, 255), font=fonts["section"])
        overall_text = _fit_text_to_lines(draw, sections["overall"], fonts["body"], 1090, 4)
        _draw_wrapped(draw, overall_text, 92, 380, 1090, fonts["body"], (186, 202, 224), line_gap=8, max_lines=4)
        cards_start_y = 545

    scores = payload["final_scores"]
    pillar_sections = sections["pillars"]
    card_w = 555
    card_h = 250 if has_snapshot else 270
    second_row_y = cards_start_y + card_h + 30
    comparisons = payload.get("personal_best_comparisons", {})
    _draw_pillar_card(draw, 60, cards_start_y, card_w, card_h, "Pressure", scores["pressure"], pillar_sections["pressure"], fonts, comparisons.get("pressure"))
    _draw_pillar_card(draw, 660, cards_start_y, card_w, card_h, "Balance", scores["balance"], pillar_sections["balance"], fonts, comparisons.get("balance"))
    _draw_pillar_card(draw, 60, second_row_y, card_w, card_h, "Rotation", scores["rotation"], pillar_sections["rotation"], fonts, comparisons.get("rotation"))
    _draw_pillar_card(draw, 660, second_row_y, card_w, card_h, "Edging", scores["edging"], pillar_sections["edging"], fonts, comparisons.get("edging"))

    improvement_y = second_row_y + card_h + 35
    draw.rounded_rectangle((60, improvement_y, 1215, 1494), radius=28, fill=(10, 24, 42), outline=(34, 54, 82), width=2)
    draw.text((92, improvement_y + 36), "Improvement Areas", fill=(238, 246, 255), font=fonts["section"])
    y = improvement_y + 90
    improvement_limit = 3 if has_snapshot else 4
    for index, item in enumerate(
        sections["improvement_areas"][:improvement_limit],
        start=1,
    ):
        draw.ellipse((96, y + 7, 116, y + 27), fill=(106, 175, 255))
        draw.text((102, y + 4), str(index), fill=(5, 10, 22), font=fonts["tiny_bold"])
        fitted_item = _fit_text_to_lines(
            draw,
            item,
            fonts["body"],
            1010,
            2,
        )
        y = _draw_wrapped(
            draw,
            fitted_item,
            136,
            y,
            1010,
            fonts["body"],
            (186, 202, 224),
            line_gap=6,
            max_lines=2,
        ) + 12

    draw.text((72, 1542), "Score scale: 60-129 Emerging | 130-169 Developing | 170-199 Proficient | 200-240 Excellent", fill=(125, 143, 166), font=fonts["small"])
    footer_parts = [f"Duration: {payload['duration_seconds']} sec"]
    if int(payload.get("turns") or 0) > 0:
        footer_parts.append(f"Turns: {payload['turns']}")
    draw.text((940, 1542), "  |  ".join(footer_parts), fill=(125, 143, 166), font=fonts["small"])

    canvas.save(report_path, "PDF", resolution=150.0)


def generate_basic_report(
    result: Dict[str, Any],
    score_timeline: List[Dict[str, Any]],
    use_openai: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a one-page styled PDF from fixed, score-band report copy.

    ``use_openai`` is retained for API compatibility. Report narratives are
    deterministic and no longer sent to or generated by an external model.
    """
    final_scores = {
        "blue_iq": _ceil_score(
            (
                result["pressure_score"]
                + result["balance_score"]
                + result["rotation_score"]
                + result["edging_score"]
            )
            / 4
        ),
        "pressure": _ceil_score(result["pressure_score"]),
        "balance": _ceil_score(result["balance_score"]),
        "rotation": _ceil_score(result["rotation_score"]),
        "edging": _ceil_score(result["edging_score"]),
    }
    score_windows = build_score_windows(score_timeline)
    payload = {
        "final_scores": final_scores,
        "approved_blueiq_level": _report_level_context(final_scores["blue_iq"]),
        "blueiq_level": _report_level_context(final_scores["blue_iq"]),
        "run_parts": _build_prompt_segments(score_windows),
        "duration_seconds": int(round(float(result.get("duration") or 0))),
        "turns": int(result.get("turns") or 0),
        "snapshot_path": result.get("snapshot_path"),
        "score_scale": "60-129 Emerging, 130-169 Developing, 170-199 Proficient, 200-240 Excellent",
        "coaching_factor_labels": COACHING_FACTOR_LABELS,
    }
    if context:
        payload.update({
            "user_name": context.get("user_name"),
            "attempt_number": context.get("attempt_number"),
            "session_date": context.get("session_date"),
            "session_number": context.get("session_number"),
        })
    payload["personal_best_comparisons"] = compare_current_scores_to_previous(
        final_scores,
        (context or {}).get("previous_personal_bests") or {},
        session_date=(context or {}).get("session_date"),
        session_number=int((context or {}).get("session_number") or 0),
        run_number=int((context or {}).get("attempt_number") or 0),
    )

    sections = _build_fixed_sections(payload)
    output_path = result["output_path"]
    report_path = os.path.splitext(output_path)[0] + "_report.pdf"
    _draw_pdf_report(report_path, payload, sections)

    report_text = "\n\n".join(
        [
            f"Overall: {_clean_report_text(sections['overall'])}",
            *[
                f"{name.title()}: {_coerce_text(section.get('summary'))} {_coerce_text(section.get('coaching_focus'))}".strip()
                for name, section in sections["pillars"].items()
            ],
            "Improvement Areas: " + " ".join(_clean_report_text(item) for item in sections["improvement_areas"]),
            "Personal Bests: " + " ".join(
                _personal_best_context_text(comparison)
                for comparison in payload["personal_best_comparisons"].values()
            ),
        ]
    )

    return {
        "report_text": report_text,
        "report_path": report_path,
        "score_windows": score_windows,
        "report_sections": sections,
        "personal_best_comparisons": payload["personal_best_comparisons"],
    }
