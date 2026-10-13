"""Automatic coach notes written to the client after each analysed run.

Notes are generated from the run's scores only. Safety does not rely on the
prompt alone: every model response passes ``validate_coach_note`` (no medical
or injury language, no technical jargon, no numbers, no prescriptive or
contradicting phrasing, short length). Anything that fails, or any API error,
falls back to a deterministic note written in the same warm voice, so a
report always carries a safe note without anyone reviewing it.
"""

import json
import logging
import os
import re
import zlib
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

PILLARS = ("pressure", "balance", "rotation", "edging")

PILLAR_NAMES = {
    "pressure": "Pressure",
    "balance": "Balance",
    "rotation": "Rotation",
    "edging": "Parallelism",
}

# Plain-language meaning of each pillar, given to the model and used by the
# fallback so neither has to invent technical descriptions.
PILLAR_PLAIN = {
    "pressure": "staying soft and springy through your legs as you turn",
    "balance": "staying centred and steady over your skis",
    "rotation": "letting your legs steer while your upper body stays calm",
    "edging": "keeping your skis side by side and moving together through each turn",
}

# Upper bound that fits the three-line note box on the PDF report.
MAX_NOTE_CHARS = 280
MIN_NOTE_CHARS = 60
MAX_SENTENCES = 4

_BANNED_PATTERNS = {
    "medical": re.compile(
        r"\b(injur\w*|pain\w*|hurt\w*|sore\w*|ache\w*|acl|mcl|ligament\w*|tendon\w*|"
        r"muscle\w*|joint\w*|doctor\w*|physio\w*|physician\w*|medic\w*|surg\w*|"
        r"sprain\w*|strain\w*|tear\w*|torn|fractur\w*|concuss\w*|rehab\w*|therap\w*|"
        r"diagnos\w*|health\w*|safe(ty)?|risk\w*|dangerous)\b",
        re.IGNORECASE,
    ),
    "jargon": re.compile(
        r"\b(angle\w*|degree\w*|flexion|extension|lateral\w*|vector\w*|pixel\w*|"
        r"frame\w*|metric\w*|algorithm\w*|data|biomechan\w*|kinetic\w*|"
        r"angulation|inclination|fore-aft|counter-rotat\w*|percent\w*|ai)\b",
        re.IGNORECASE,
    ),
    "prescriptive": re.compile(
        r"\b(must|never|always|exactly|drill\w*|exercise\w*|reps?|sets? of|"
        r"you need to|you have to|make sure)\b|%",
        re.IGNORECASE,
    ),
    "contradiction": re.compile(
        r"\b(ignore\w*|disregard\w*|contrary|disagree\w*|instead of what|"
        r"wrong|incorrect\w*|bad|poor\w*|terrible|fail\w*)\b",
        re.IGNORECASE,
    ),
    "format": re.compile(r"\d|https?://|[#*_`<>{}\[\]]"),
}

SYSTEM_PROMPT = """You are a friendly ski coach at Bluerun. After each run is analysed on video, you write a short personal note to the skier that appears on their report.

Voice:
- Warm, encouraging and human, like a coach chatting at the bottom of the run.
- Plain everyday words and short sentences. Talk to the skier as "you".

Content:
- Base the note only on the scores provided. Mention the strongest area and one area to keep building, in general terms.
- You may use the pillar names Pressure, Balance, Rotation and Parallelism, described the plain way given in the data.
- If there is a new personal best, celebrate it briefly.
- Keep advice general and encouraging. Do not give drills, exercises, step-by-step instructions or exact technique corrections.

Never:
- Never mention injury, pain, the body's health, safety or anything medical.
- Never contradict, question or replace what an instructor may have told the skier.
- Never use numbers, scores, percentages or technical terms such as angles, degrees, flexion, frames or data.
- Never use "must", "always", "never", "you need to" or similar commands.
- Never criticise; frame anything to improve as the next thing to enjoy working on.

Length: 2 to 4 short sentences, under 250 characters in total.

Return only JSON in the form {"note": "..."}."""


def _ai_enabled() -> bool:
    return os.getenv("AI_COACH_NOTES_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API")


def _int_score(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


_TYPOGRAPHIC = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _clean(text: Any) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").translate(_TYPOGRAPHIC)).strip()
    return cleaned.strip("\"'“”‘’ ").strip()


def validate_coach_note(text: str) -> Optional[str]:
    """Return the reason a note is unsafe or unusable, or None when it is fine."""
    note = _clean(text)
    if len(note) < MIN_NOTE_CHARS:
        return "too_short"
    if len(note) > MAX_NOTE_CHARS:
        return "too_long"
    sentences = [part for part in re.split(r"[.!?]+", note) if part.strip()]
    if len(sentences) > MAX_SENTENCES:
        return "too_many_sentences"
    for reason, pattern in _BANNED_PATTERNS.items():
        if pattern.search(note):
            return reason
    return None


def _ranked(scores: Mapping[str, Any]):
    pillar_scores = [(name, _int_score(scores.get(name))) for name in PILLARS]
    ranked = sorted(pillar_scores, key=lambda item: (-item[1], PILLARS.index(item[0])))
    return ranked[0][0], ranked[-1][0], ranked[0][1] == ranked[-1][1]


def _new_personal_bests(comparisons: Optional[Mapping[str, Any]]):
    found = []
    for metric, comparison in (comparisons or {}).items():
        status = comparison.get("status") if isinstance(comparison, Mapping) else None
        if status == "new_personal_best":
            found.append(metric)
    return found


_OPENERS = (
    "Nice work out there today.",
    "That was a solid run.",
    "Good skiing today.",
    "Great to see you out on the hill.",
)

_STRENGTH_LINES = (
    "Your {best} stood out, so keep enjoying that feeling.",
    "{best} was your strongest area on this run.",
    "You looked most at home with your {best}.",
)

_FOCUS_LINES = (
    "Next time, give a little extra attention to {focus_name}, {focus_plain}.",
    "A nice next step is {focus_name}: {focus_plain}.",
    "Keep building your {focus_name} by {focus_plain}.",
)

_EVEN_LINES = (
    "All four areas are coming along together, which is a great base to build on.",
    "Your skiing looks nicely even across the board, so keep that rhythm going.",
)

_PB_OPENERS = (
    "Great run, and a new personal best too.",
    "A new personal best today, nice work.",
)


def fallback_coach_note(
    scores: Mapping[str, Any],
    comparisons: Optional[Mapping[str, Any]] = None,
) -> str:
    """Deterministic note in the coach voice, varied per run."""
    best, focus, all_even = _ranked(scores)
    seed = zlib.crc32(
        json.dumps({name: _int_score(scores.get(name)) for name in PILLARS}, sort_keys=True).encode("utf-8")
    )

    opener_pool = _PB_OPENERS if _new_personal_bests(comparisons) else _OPENERS
    opener = opener_pool[seed % len(opener_pool)]
    if all_even:
        return f"{opener} {_EVEN_LINES[seed % len(_EVEN_LINES)]}"

    strength = _STRENGTH_LINES[(seed >> 3) % len(_STRENGTH_LINES)].format(best=PILLAR_NAMES[best])
    if strength[0].islower():
        strength = strength[0].upper() + strength[1:]
    focus_line = _FOCUS_LINES[(seed >> 6) % len(_FOCUS_LINES)].format(
        focus_name=PILLAR_NAMES[focus],
        focus_plain=PILLAR_PLAIN[focus],
    )
    note = f"{opener} {strength} {focus_line}"
    if len(note) > MAX_NOTE_CHARS:
        note = f"{strength} {focus_line}"
    return note


def _prompt_payload(scores: Mapping[str, Any], comparisons: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    best, focus, all_even = _ranked(scores)
    return {
        "strongest_area": PILLAR_NAMES[best],
        "area_to_keep_building": None if all_even else PILLAR_NAMES[focus],
        "all_areas_even": all_even,
        "plain_meaning": {PILLAR_NAMES[name]: PILLAR_PLAIN[name] for name in PILLARS},
        "relative_scores_high_to_low": [
            PILLAR_NAMES[name]
            for name, _ in sorted(
                ((name, _int_score(scores.get(name))) for name in PILLARS),
                key=lambda item: -item[1],
            )
        ],
        "new_personal_best_areas": [
            PILLAR_NAMES.get(metric, "Blue IQ") for metric in _new_personal_bests(comparisons)
        ],
    }


def _request_ai_note(payload: Dict[str, Any], api_key: str) -> str:
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Run data:\n{json.dumps(payload, indent=2)}"},
            ],
            "temperature": 0.7,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _clean(json.loads(content).get("note"))


def generate_coach_note(
    scores: Mapping[str, Any],
    comparisons: Optional[Mapping[str, Any]] = None,
    attempts: int = 2,
) -> Tuple[str, str]:
    """Return ``(note, source)`` where source is ``"ai"`` or ``"template"``."""
    api_key = _api_key()
    if _ai_enabled() and api_key:
        payload = _prompt_payload(scores, comparisons)
        for attempt in range(1, attempts + 1):
            try:
                note = _request_ai_note(payload, api_key)
            except Exception as exc:
                logger.warning("AI coach note request failed (attempt %s): %s", attempt, exc)
                break
            problem = validate_coach_note(note)
            if problem is None:
                return note, "ai"
            logger.warning("AI coach note rejected by guardrails (%s): %r", problem, note)

    return fallback_coach_note(scores, comparisons), "template"
