import json
import logging
import os
from datetime import date, datetime

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

logger = logging.getLogger(__name__)

MAX_BLUE_SCORE = 240
LEVEL_COUNT = 9
SCORE_BANDS = (
    {"key": "emerging", "label": "Emerging", "min": 60, "max": 129},
    {"key": "developing", "label": "Developing", "min": 130, "max": 169},
    {"key": "proficient", "label": "Proficient", "min": 170, "max": 199},
    {"key": "excellent", "label": "Excellent", "min": 200, "max": 240},
)
BLUE_IQ_LEVELS = (
    {"level": 1, "label": "Beginner \u2022 Level 1", "min": 60, "max": 80},
    {"level": 2, "label": "Beginner \u2022 Level 2", "min": 81, "max": 100},
    {"level": 3, "label": "Intermediate \u2022 Level 3", "min": 101, "max": 120},
    {"level": 4, "label": "Intermediate \u2022 Level 4", "min": 121, "max": 140},
    {"level": 5, "label": "Intermediate \u2022 Level 5", "min": 141, "max": 160},
    {"level": 6, "label": "Intermediate \u2022 Level 6", "min": 161, "max": 180},
    {"level": 7, "label": "Intermediate \u2022 Level 7", "min": 181, "max": 200},
    {"level": 8, "label": "Expert \u2022 Level 8", "min": 201, "max": 220},
    {"level": 9, "label": "Expert \u2022 Level 9", "min": 221, "max": 240},
)


def draw_logo_placeholder(overlay, x, y, w, h, border_color, text_color):
    """Draw a placeholder when logo is not available"""
    logger.debug(f"Drawing logo placeholder at position ({x}, {y}) with dimensions {w}x{h}")
    try:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), border_color, 2)
        cv2.putText(overlay, "LOGO", (x + w//2 - 25, y + h//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        logger.debug("Logo placeholder drawn successfully")
    except Exception as e:
        logger.error(f"Error drawing logo placeholder: {e}")


def _score_tier(score):
    band = _score_band(score)
    color_map = {
        "emerging": (74, 96, 214),
        "developing": (42, 157, 244),
        "proficient": (108, 166, 32),
        "excellent": (42, 42, 42),
    }
    return band["label"], color_map[band["key"]]


def _score_band(score):
    score = int(round(_clamp_score(score)))
    for band in SCORE_BANDS:
        if band["min"] <= score <= band["max"]:
            return band
    return SCORE_BANDS[0] if score < SCORE_BANDS[0]["min"] else SCORE_BANDS[-1]


def _clamp_score(score, min_score=60, max_score=240):
    return max(min_score, min(max_score, float(score or 0)))


def _score_fill_ratio(score):
    return max(0.0, min(1.0, float(score or 0) / MAX_BLUE_SCORE))


def _draw_metric_bar(canvas, label, score, x, y, width, colors, highlight=False):
    score = _clamp_score(score)
    tier_label, tier_color = _score_tier(score)
    bar_x = x + 168
    bar_y = y + 16
    bar_w = width - 265
    bar_h = 18
    fill_w = int(bar_w * _score_fill_ratio(score))

    label_color = colors["accent"] if highlight else colors["text"]
    cv2.putText(canvas, label, (x, y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.68, label_color, 2)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), colors["bar_bg"], -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), tier_color, -1)

    for boundary in (100, 150, 200):
        marker_x = bar_x + int(bar_w * (boundary / MAX_BLUE_SCORE))
        cv2.line(canvas, (marker_x, bar_y), (marker_x, bar_y + bar_h), colors["panel"], 2)

    cv2.circle(canvas, (bar_x + fill_w, bar_y + bar_h // 2), 5, colors["panel"], -1)
    cv2.circle(canvas, (bar_x + fill_w, bar_y + bar_h // 2), 5, tier_color, 2)
    cv2.putText(canvas, f"{score:.0f}", (bar_x + bar_w + 18, y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.68, label_color, 2)
    cv2.putText(canvas, tier_label.upper(), (bar_x, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, colors["text"], 2)


def _latest_metric_value(metrics, key, precision=1):
    value = metrics.get(key)
    if isinstance(value, list):
        value = value[-1] if value else None
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)


def _blueiq_level_label(score):
    return _blueiq_level_info(score)["label"]


def _blueiq_level_info(score):
    score = int(round(_clamp_score(score)))
    for level in BLUE_IQ_LEVELS:
        if level["min"] <= score <= level["max"]:
            return level
    return BLUE_IQ_LEVELS[0] if score < BLUE_IQ_LEVELS[0]["min"] else BLUE_IQ_LEVELS[-1]


def get_unique_output_path(outputs_dir, base_name, display_mode, extension=".mp4", variant=None):
    variant_suffix = f"_{variant}" if variant else ""
    output_name = f"output_{base_name}_{display_mode}{variant_suffix}{extension}"
    output_path = os.path.join(outputs_dir, output_name)
    if not os.path.exists(output_path):
        return output_path

    counter = 1
    while True:
        output_name = f"output_{base_name}_{display_mode}{variant_suffix}({counter}){extension}"
        output_path = os.path.join(outputs_dir, output_name)
        if not os.path.exists(output_path):
            return output_path
        counter += 1


def write_react_overlay_page(output_path, result_data, display_mode):
    html_path = os.path.splitext(output_path)[0] + ".html"
    video_name = os.path.basename(output_path)
    payload = dict(result_data)
    payload["video_src"] = video_name
    payload["blue_iq"] = (
        payload["pressure_score"] +
        payload["balance_score"] +
        payload["rotation_score"] +
        payload["edging_score"]
    ) / 4
    payload["level_label"] = _blueiq_level_label(payload["blue_iq"])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bluerun Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --navy-950: #060a14;
      --navy-900: #0a0e1a;
      --navy-800: #0d1525;
      --navy-700: #111c30;
      --navy-600: #162038;
      --navy-500: #1e2d4a;
      --card-bg: rgba(255,255,255,0.04);
      --card-border: rgba(255,255,255,0.08);
      --card-border-hover: rgba(255,255,255,0.14);
      --blue-400: #6aafff;
      --blue-300: #93c5fd;
      --blue-500: #3b82f6;
      --green-400: #4ade80;
      --green-500: #22c55e;
      --green-600: #16a34a;
      --amber-400: #fbbf24;
      --amber-500: #f59e0b;
      --amber-600: #d97706;
      --red-400: #f87171;
      --red-500: #ef4444;
      --red-600: #dc2626;
      --text-primary: rgba(255,255,255,0.92);
      --text-secondary: rgba(255,255,255,0.5);
      --text-tertiary: rgba(255,255,255,0.28);
      --font-sans: 'DM Sans', system-ui, sans-serif;
      --font-mono: 'DM Mono', monospace;
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 16px;
      --radius-xl: 20px;
      --radius-pill: 999px;
    }}
    html, body {{ height: 100%; }}
    body {{
      font-family: var(--font-sans);
      background: #060a14;
      color: var(--text-primary);
      display: grid;
      place-items: center;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}
    .shell {{
      width: min(1600px, 100vw);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      min-height: 100vh;
      background: var(--navy-900);
    }}
    .video-pane {{
      position: relative;
      background: #000;
      display: flex;
      align-items: center;
    }}
    .video-pane video {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
    }}
    .sidebar {{
      background: linear-gradient(175deg, var(--navy-900) 0%, var(--navy-800) 55%, #0a1020 100%);
      border-left: 1px solid rgba(255,255,255,0.06);
      padding: 28px 24px 32px;
      display: flex;
      flex-direction: column;
      gap: 0;
      overflow-y: auto;
      position: relative;
    }}
    .sidebar::before {{
      content: '';
      position: absolute;
      top: -100px; left: -100px;
      width: 360px; height: 360px;
      background: radial-gradient(circle, rgba(106,175,255,0.05) 0%, transparent 70%);
      pointer-events: none;
    }}
    .logo-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 26px;
    }}
    .logo-svg {{ height: 32px; width: auto; }}
    .coach-badge {{
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--blue-400);
      background: rgba(106,175,255,0.10);
      border: 1px solid rgba(106,175,255,0.22);
      padding: 5px 12px;
      border-radius: var(--radius-pill);
    }}
    .athlete-badge {{
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--green-400);
      background: rgba(74,222,128,0.10);
      border: 1px solid rgba(74,222,128,0.22);
      padding: 5px 12px;
      border-radius: var(--radius-pill);
    }}
    .section-header {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .section-icon {{
      width: 26px; height: 26px;
      background: rgba(106,175,255,0.09);
      border: 1px solid rgba(106,175,255,0.18);
      border-radius: 7px;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px;
      flex-shrink: 0;
    }}
    .section-title {{
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-secondary);
    }}
    .iq-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 18px 20px;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .iq-label {{
      font-size: 13px;
      font-weight: 600;
      color: var(--blue-400);
      margin-bottom: 3px;
    }}
    .iq-sublabel {{
      font-size: 11px;
      color: rgba(255,255,255,0.55);
      font-weight: 400;
    }}
    .iq-score-block {{ text-align: right; }}
    .iq-score {{
      font-family: var(--font-mono);
      font-size: 52px;
      font-weight: 500;
      color: #ffffff;
      letter-spacing: -1px;
      line-height: 1;
    }}
    .iq-tier-label {{
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      text-align: right;
      margin-top: 3px;
    }}
    .tier-excellent {{ color: var(--blue-400); }}
    .tier-strong {{ color: var(--green-400); }}
    .tier-developing {{ color: var(--amber-400); }}
    .tier-needs {{ color: var(--red-400); }}
    .level-pills {{
      display: flex;
      gap: 4px;
      margin-bottom: 22px;
    }}
    .pill {{
      height: 4px;
      border-radius: var(--radius-pill);
      flex: 1;
    }}
    .pill-needs {{ background: var(--red-500); opacity: 0.5; }}
    .pill-developing {{ background: var(--amber-500); opacity: 0.5; }}
    .pill-strong {{ background: var(--green-500); opacity: 0.5; }}
    .pill-excellent {{ background: var(--blue-400); flex: 2; }}
    .metrics-stack {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 22px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 14px 16px 14px 19px;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s;
    }}
    .metric-card:hover {{ border-color: var(--card-border-hover); }}
    .metric-card::before {{
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 3px;
      border-radius: 3px 0 0 3px;
    }}
    .mc-excellent::before {{ background: var(--blue-400); }}
    .mc-strong::before {{ background: var(--green-500); }}
    .mc-developing::before {{ background: var(--amber-500); }}
    .mc-needs::before {{ background: var(--red-500); }}
    .metric-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }}
    .metric-name {{
      font-size: 13px;
      font-weight: 500;
      color: rgba(255,255,255,0.8);
    }}
    .metric-right {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .metric-score {{
      font-family: var(--font-mono);
      font-size: 19px;
      font-weight: 500;
      color: #ffffff;
    }}
    .metric-badge {{
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: var(--radius-pill);
    }}
    .mb-excellent {{ background: rgba(106,175,255,0.10); color: var(--blue-400); border: 1px solid rgba(106,175,255,0.22); }}
    .mb-strong {{ background: rgba(34,197,94,0.10); color: var(--green-400); border: 1px solid rgba(34,197,94,0.22); }}
    .mb-developing {{ background: rgba(245,158,11,0.10); color: var(--amber-400); border: 1px solid rgba(245,158,11,0.22); }}
    .mb-needs {{ background: rgba(239,68,68,0.10); color: var(--red-400); border: 1px solid rgba(239,68,68,0.22); }}
    .progress-track {{
      height: 5px;
      background: rgba(255,255,255,0.07);
      border-radius: var(--radius-pill);
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      border-radius: var(--radius-pill);
      transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .pf-excellent {{ background: linear-gradient(90deg, var(--blue-500), var(--blue-400)); }}
    .pf-strong    {{ background: linear-gradient(90deg, var(--green-600), var(--green-500)); }}
    .pf-developing {{ background: linear-gradient(90deg, var(--amber-600), var(--amber-500)); }}
    .pf-needs     {{ background: linear-gradient(90deg, var(--red-600), var(--red-500)); }}
    .divider {{
      height: 1px;
      background: rgba(255,255,255,0.06);
      margin: 18px 0;
    }}
    .run-stats {{ display: flex; flex-direction: column; gap: 10px; margin-top: 14px; }}
    .run-stat-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: var(--radius-sm);
    }}
    .run-stat-label {{ font-size: 13px; color: var(--text-secondary); font-weight: 400; }}
    .run-stat-value {{ font-family: var(--font-mono); font-size: 14px; color: var(--blue-400); font-weight: 500; }}
    .sidebar::-webkit-scrollbar {{ width: 4px; }}
    .sidebar::-webkit-scrollbar-track {{ background: transparent; }}
    .sidebar::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 2px; }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    const data = {json.dumps(payload)};
    const e = React.createElement;
    function tierKey(score) {{
      if (score >= 200) return 'excellent';
      if (score >= 150) return 'strong';
      if (score >= 100) return 'developing';
      return 'needs';
    }}
    function tierLabel(score) {{
      return {{ excellent: 'Excellent', strong: 'Strong', developing: 'Developing', needs: 'Needs Work' }}[tierKey(score)];
    }}
    function pct(score) {{
      return Math.max(0, Math.min(100, ((score - 60) / 180) * 100));
    }}
    function LogoSVG() {{
      return e('svg', {{ className:'logo-svg', viewBox:'0 0 148 36', fill:'none', xmlns:'http://www.w3.org/2000/svg' }},
        e('polygon', {{ points:'14,2 22,2 14,10 6,10', fill:'#6aafff', opacity:'0.9' }}),
        e('polygon', {{ points:'6,10 14,10 6,18 0,12', fill:'#6aafff', opacity:'0.65' }}),
        e('polygon', {{ points:'10,18 18,18 26,26 18,26', fill:'#6aafff', opacity:'0.9' }}),
        e('polygon', {{ points:'18,26 26,26 18,34 10,34', fill:'#6aafff', opacity:'0.65' }}),
        e('text', {{ x:'35', y:'26', fill:'#6aafff', fontSize:'22', fontWeight:'500',
          fontFamily:"'DM Sans', sans-serif", letterSpacing:'-0.3' }}, 'bluerun')
      );
    }}
    function MetricCard({{ name, score }}) {{
      const tk = tierKey(score);
      const tl = tierLabel(score);
      const width = pct(score);
      return e('div', {{ className: `metric-card mc-${{tk}}` }},
        e('div', {{ className: 'metric-top' }},
          e('span', {{ className: 'metric-name' }}, name),
          e('div', {{ className: 'metric-right' }},
            e('span', {{ className: 'metric-score' }}, Math.round(score)),
            e('span', {{ className: `metric-badge mb-${{tk}}` }}, tl)
          )
        ),
        e('div', {{ className: 'progress-track' }},
          e('div', {{ className: `progress-fill pf-${{tk}}`, style: {{ width: width + '%' }} }})
        )
      );
    }}
    function App() {{
      const coach = data.display_mode === 'coach';
      const iqTk = tierKey(data.blue_iq);
      return e('main', {{ className: 'shell' }},
        e('div', {{ className: 'video-pane' }},
          e('video', {{ src: data.video_src, controls: true, autoPlay: false }})
        ),
        e('aside', {{ className: 'sidebar' }},
          e('div', {{ className: 'logo-row' }},
            e(LogoSVG),
            e('span', {{ className: coach ? 'coach-badge' : 'athlete-badge' }}, data.display_mode.toUpperCase())
          ),
          e('div', {{ className: 'section-header' }},
            e('div', {{ className: 'section-icon' }}, '\ud83c\udfaf'),
            e('span', {{ className: 'section-title' }}, 'Performance Profile')
          ),
          e('div', {{ className: 'iq-card' }},
            e('div', null,
              e('div', {{ className: 'iq-label' }}, 'Blue IQ'),
              e('div', {{ className: 'iq-sublabel' }}, data.level_label)
            ),
            e('div', {{ className: 'iq-score-block' }},
              e('div', {{ className: 'iq-score' }}, Math.round(data.blue_iq)),
              e('div', {{ className: `iq-tier-label tier-${{iqTk}}` }}, tierLabel(data.blue_iq).toUpperCase())
            )
          ),
          e('div', {{ className: 'level-pills' }},
            e('div', {{ className: 'pill pill-needs' }}),
            e('div', {{ className: 'pill pill-developing' }}),
            e('div', {{ className: 'pill pill-strong' }}),
            e('div', {{ className: 'pill pill-excellent' }})
          ),
          e('div', {{ className: 'metrics-stack' }},
            e(MetricCard, {{ name: 'Pressure', score: data.pressure_score }}),
            e(MetricCard, {{ name: 'Balance',  score: data.balance_score }}),
            e(MetricCard, {{ name: 'Rotation', score: data.rotation_score }}),
            e(MetricCard, {{ name: 'Edging',   score: data.edging_score }})
          ),
          e('div', {{ className: 'divider' }}),
          e('div', {{ className: 'section-header' }},
            e('div', {{ className: 'section-icon' }}, '\ud83d\udce1'),
            e('span', {{ className: 'section-title' }}, coach ? 'Live Run Metrics' : 'This Run')
          ),
          e('div', {{ className: 'run-stats' }},
            e('div', {{ className: 'run-stat-row' }},
              e('span', {{ className: 'run-stat-label' }}, 'Turns completed'),
              e('span', {{ className: 'run-stat-value' }}, data.turns)
            ),
            e('div', {{ className: 'run-stat-row' }},
              e('span', {{ className: 'run-stat-label' }}, 'Speed'),
              e('span', {{ className: 'run-stat-value' }}, (data.speed != null ? (+data.speed).toFixed(1) + ' px/frame' : 'N/A'))
            )
          )
        )
      );
    }}
    ReactDOM.createRoot(document.getElementById('root')).render(e(App));
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def _rgb(color_bgr):
    return (color_bgr[2], color_bgr[1], color_bgr[0])


def _load_font(size, bold=False):
    if ImageFont is None:
        return None

    candidates = []
    if bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "arialbd.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "arial.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_session_font(size):
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


def _draw_video_session_label_pillow(draw, athlete_name, attempt_number=None, session_date=None):
    if not athlete_name:
        return

    name_font = _load_session_font(30)
    meta_font = _load_session_font(12)
    x = 24
    name_y = 22
    name = str(athlete_name).strip()

    draw.text((x + 2, name_y + 2), name, fill=(0, 0, 0), font=name_font)
    draw.text((x, name_y), name, fill=(255, 255, 255), font=name_font)

    bbox = draw.textbbox((x, name_y), name, font=name_font)
    name_baseline = bbox[3]
    line_y = name_baseline + 6
    draw.rectangle((x, line_y, x + 36, line_y + 3), fill=(106, 175, 255))

    meta_y = line_y + 13
    cursor_x = x
    if attempt_number:
        attempt_text = f"RUN {attempt_number}"
        draw.text((cursor_x, meta_y), attempt_text, fill=(106, 175, 255), font=meta_font)
        cursor_x += int(draw.textlength(attempt_text, font=meta_font)) + 10

    formatted_date = _format_session_date(session_date)
    if attempt_number and formatted_date:
        dot_cx = cursor_x + 2
        dot_cy = meta_y + 8
        draw.ellipse((dot_cx - 2, dot_cy - 2, dot_cx + 2, dot_cy + 2), fill=(100, 110, 120))
        cursor_x += 12

    if formatted_date:
        draw.text((cursor_x, meta_y), formatted_date, fill=(160, 175, 190), font=meta_font)


def _premium_metric_bar(draw, label, score, x, y, width, colors, fonts):
    score = _clamp_score(score)
    tier_label, tier_color_bgr = _score_tier(score)
    tier_color = _rgb(tier_color_bgr)
    bar_x = x + 165
    bar_y = y + 21
    bar_w = width - 250
    bar_h = 15
    fill_w = int(bar_w * ((score - 60) / 180))

    draw.text((x, y + 11), label, fill=colors["text"], font=fonts["metric"])
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=8, fill=colors["bar_bg"])
    draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=8, fill=tier_color)
    for boundary in (100, 150, 200):
        marker_x = bar_x + int(bar_w * ((boundary - 60) / 180))
        draw.line((marker_x, bar_y, marker_x, bar_y + bar_h), fill=colors["panel"], width=2)

    draw.ellipse((bar_x + fill_w - 5, bar_y + 2, bar_x + fill_w + 5, bar_y + 12), fill=colors["panel"], outline=tier_color, width=2)
    draw.text((bar_x + bar_w + 16, y + 8), f"{score:.0f}", fill=colors["text"], font=fonts["metric"])
    draw.text((bar_x, y + 42), tier_label.upper(), fill=colors["muted_dark"], font=fonts["tiny"])


def create_premium_overlay(
    frame,
    metrics,
    frame_number,
    TARGET_WIDTH,
    logo_path=None,
    display_mode="coach",
    athlete_name=None,
    attempt_number=None,
    session_date=None,
    user_name=None,
):
    """Render a dark modern burned-in sidebar using Pillow.

    Layout (720px sidebar, 720px frame height):
      - Logo + badge header
      - Performance Profile section header
      - Blue IQ card (score + tier tag right-aligned)
      - Level pills
      - Four metric cards (Pressure, Balance, Rotation, Edging)
        Each: label left | score+tag right | progress bar bottom
      - Divider
      - This Run / Live Run Metrics section header
      - Two stat rows: label left | value right (inline, not stacked)
    """
    if Image is None:
        raise RuntimeError("Pillow is required for premium overlay rendering")

    sidebar_width = 460
    frame_height  = frame.shape[0]   # expected 720
    video_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_img     = Image.fromarray(video_rgb)

    canvas = Image.new("RGB", (TARGET_WIDTH + sidebar_width, frame_height), (10, 14, 26))
    canvas.paste(video_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_video_session_label_pillow(draw, athlete_name or user_name, attempt_number, session_date)

    # ── COLOR PALETTE ────────────────────────────────────────────────────────
    C = {
        "bg_top":      (10,  14,  26),
        "bg_mid":      (13,  21,  37),
        "bg_bottom":   (10,  16,  32),
        "card":        (18,  27,  46),
        "card_soft":   (17,  26,  44),
        "border":      (42,  54,  80),
        "text":        (238, 246, 255),
        "text_soft":   (220, 230, 242),
        "muted":       (172, 184, 202),
        "faint":       (130, 144, 166),
        "track":       (43,  53,  74),
        "blue":        (106, 175, 255),
        "green":       (34,  197,  94),
        "amber":       (245, 158,  11),
        "red":         (239,  68,  68),
    }

    # ── FONTS ────────────────────────────────────────────────────────────────
    F = {
        "brand":        _load_font(28, bold=True),
        "section":      _load_font(10, bold=True),
        "badge":        _load_font(11, bold=True),
        "iq_label":     _load_font(17, bold=True),
        "iq_sublabel":  _load_font(13, bold=False),
        "iq_score":     _load_font(52, bold=True),
        "score_den":    _load_font(19, bold=True),
        "iq_tier":      _load_font(12, bold=True),
        "metric_name":  _load_font(14, bold=False),
        "metric_score": _load_font(22, bold=True),
        "metric_den":   _load_font(13, bold=True),
        "metric_tag":   _load_font(11, bold=True),
        "tiny":         _load_font(10, bold=True),
        "stat_label":   _load_font(13, bold=False),
        "stat_value":   _load_font(15, bold=True),
    }

    sidebar_x = TARGET_WIDTH

    # ── BACKGROUND GRADIENT ──────────────────────────────────────────────────
    for y_px in range(frame_height):
        ratio = y_px / max(1, frame_height - 1)
        if ratio < 0.45:
            t = ratio / 0.45
            bg = tuple(int(C["bg_top"][i]*(1-t) + C["bg_mid"][i]*t) for i in range(3))
        else:
            t = (ratio - 0.45) / 0.55
            bg = tuple(int(C["bg_mid"][i]*(1-t) + C["bg_bottom"][i]*t) for i in range(3))
        draw.line((sidebar_x, y_px, sidebar_x + sidebar_width, y_px), fill=bg)

    # Subtle radial glow behind logo area
    for radius, shade in ((360, (13,31,58)), (240,(11,26,50)), (140,(10,22,42))):
        draw.ellipse((sidebar_x, -radius//3, sidebar_x+radius, radius), fill=shade)

    # Left border separator
    draw.line((sidebar_x, 0, sidebar_x, frame_height), fill=(30, 42, 64), width=1)

    # ── LAYOUT CONSTANTS ─────────────────────────────────────────────────────
    MARGIN   = 20
    panel_x  = sidebar_x + MARGIN
    panel_w  = sidebar_width - MARGIN * 2

    # Fixed column anchors for metric rows
    LABEL_X        = panel_x + 16
    TAG_W          = 88
    TAG_H          = 22
    TAG_X          = panel_x + panel_w - TAG_W - 10
    SCORE_RIGHT_X  = TAG_X - 10   # score right-edge

    # ── TIER HELPERS ─────────────────────────────────────────────────────────
    def _band(s):
        return _score_band(s)

    def _tcolor(s):
        return {
            "excellent": C["blue"],
            "proficient": C["green"],
            "developing": C["amber"],
            "emerging": C["red"],
        }[_band(s)["key"]]

    def _tlabel(s):
        return _band(s)["label"]

    def _level_number(score):
        return _blueiq_level_info(score)["level"]

    def _level_color(level):
        if level <= 5:
            return C["amber"]
        if level <= 7:
            return C["green"]
        return C["blue"]

    def draw_level_dots(x, y, level, active_color):
        dot_gap = 7
        for idx in range(LEVEL_COUNT):
            cx = x + idx * dot_gap
            fill = active_color if idx < level else (46, 61, 86)
            outline = active_color if idx < level else (72, 88, 112)
            draw.ellipse((cx, y, cx + 4, y + 4), fill=fill, outline=outline)

    # Compute Blue IQ
    blue_iq = (
        metrics["Blue_pressure_final"] + metrics["Blue_rotation_final"] +
        metrics["Blue_edging_final"]   + metrics["Blue_balance_final"]
    ) / 4

    # ── SECTION: LOGO + BADGE ────────────────────────────────────────────────
    y = 16

    # Logo — diamond mark then wordmark
    logo_loaded = False
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo.thumbnail((200, 36), Image.LANCZOS)
            canvas.paste(logo, (panel_x, y + 2), logo)
            logo_loaded = True
        except Exception as exc:
            logger.error(f"Premium logo load error: {exc}")

    if not logo_loaded:
        mx, my = panel_x, y + 2
        # Diamond icon (4 rhombus facets)
        for idx, poly in enumerate([
            [(mx+12,my+0), (mx+20,my+0), (mx+12,my+8), (mx+4,my+8)],
            [(mx+4,my+8),  (mx+12,my+8), (mx+4,my+16), (mx+0,my+10)],
            [(mx+8,my+16), (mx+16,my+16),(mx+24,my+24),(mx+16,my+24)],
            [(mx+16,my+24),(mx+24,my+24),(mx+16,my+32),(mx+8,my+32)],
        ]):
            draw.polygon(poly, fill=C["blue"] if idx % 2 == 0 else (82, 148, 222))
        draw.text((panel_x + 32, y + 3), "bluerun", fill=C["blue"], font=F["brand"])

    # Athlete / Coach badge (top-right)
    badge_label = "COACH" if display_mode == "coach" else "ATHLETE"
    badge_color = C["blue"] if display_mode == "coach" else C["green"]
    BW, BH = 88, 26
    bx = sidebar_x + sidebar_width - MARGIN - BW
    by = y + 5
    draw.rounded_rectangle((bx, by, bx+BW, by+BH), radius=13, fill=(18,32,56), outline=badge_color, width=1)
    btw = draw.textlength(badge_label, font=F["badge"])
    draw.text((bx + (BW - btw) / 2, by + 6), badge_label, fill=badge_color, font=F["badge"])

    y += 48   # breathing room after logo row

    # ── SECTION HEADER HELPER ────────────────────────────────────────────────
    def section_header(sy, title, live=False):
        # Small circle icon
        draw.rounded_rectangle((panel_x, sy, panel_x+22, sy+22),
                                radius=6, fill=(18,32,56), outline=(38,58,86), width=1)
        draw.ellipse((panel_x+6, sy+6, panel_x+16, sy+16), outline=C["blue"], width=2)
        draw.text((panel_x + 28, sy + 5), title, fill=C["muted"], font=F["section"])
        if live:
            dx = panel_x + panel_w - 50
            draw.ellipse((dx, sy+7, dx+7, sy+14), fill=C["red"])
            draw.text((dx+11, sy+5), "LIVE", fill=C["red"], font=F["tiny"])
        return sy + 28

    # ── PERFORMANCE PROFILE ──────────────────────────────────────────────────
    y = section_header(y, "PERFORMANCE PROFILE")

    # Blue IQ card
    IQ_H = 74
    draw.rounded_rectangle((panel_x, y, panel_x+panel_w, y+IQ_H),
                            radius=14, fill=C["card"], outline=C["border"], width=1)
    draw.text((panel_x+16, y+10), "Blue IQ", fill=C["blue"], font=F["iq_label"])
    level_label = _blueiq_level_label(blue_iq)
    draw.text((panel_x+16, y+36), level_label, fill=C["muted"], font=F["iq_sublabel"])

    display_blue_iq = int(round(_clamp_score(blue_iq)))
    score_str = f"{display_blue_iq}"
    sw = draw.textlength(score_str, font=F["iq_score"])
    denom = f"/{MAX_BLUE_SCORE}"
    denom_w = draw.textlength(denom, font=F["score_den"])
    tier_txt  = _tlabel(display_blue_iq).upper()
    tier_col  = _tcolor(display_blue_iq)

    iq_tag_w = max(90, int(draw.textlength(tier_txt, font=F["iq_tier"]) + 22))
    iq_tag_h = 24
    iq_tag_x = panel_x + panel_w - iq_tag_w - 12
    iq_tag_y = y + 28

    score_right = iq_tag_x - 12
    denom_x = score_right - denom_w
    draw.text((denom_x - sw - 3, y + 4), score_str, fill=C["text"], font=F["iq_score"])
    draw.text((denom_x, y + 39), denom, fill=C["muted"], font=F["score_den"])
    draw.rounded_rectangle((iq_tag_x, iq_tag_y, iq_tag_x+iq_tag_w, iq_tag_y+iq_tag_h),
                            radius=11, fill=(20,36,58), outline=tier_col, width=1)
    tw = draw.textlength(tier_txt, font=F["iq_tier"])
    draw.text((iq_tag_x + (iq_tag_w - tw)/2, iq_tag_y + 5), tier_txt, fill=tier_col, font=F["iq_tier"])

    y += IQ_H + 8

    # 9-level bar: each completed level remains visually separated.
    level_bar_x = panel_x
    level_bar_y = y
    level_bar_w = panel_w
    level_bar_h = 6
    segment_gap = 4
    segment_w = (level_bar_w - segment_gap * (LEVEL_COUNT - 1)) / LEVEL_COUNT
    completed_levels = _level_number(display_blue_iq)
    fill_color = _level_color(completed_levels)
    for idx in range(LEVEL_COUNT):
        sx = int(level_bar_x + idx * (segment_w + segment_gap))
        ex = int(level_bar_x + (idx + 1) * segment_w + idx * segment_gap)
        segment_color = fill_color if idx < completed_levels else C["track"]
        draw.rounded_rectangle((sx, level_bar_y, ex, level_bar_y + level_bar_h),
                                radius=3, fill=segment_color)
    y += 16

    # ── METRIC CARDS ─────────────────────────────────────────────────────────
    # Budget: allocate card height so everything fits top-to-bottom with safe margin.
    # Fixed costs below metric cards:
    #   divider_y_step (6) + divider_line (1) + divider_h (18) = ~25
    #   section header = 28
    #   2 stat cards + 1 gap
    CARD_GAP  = 6
    STAT_H    = 46
    STAT_GAP  = 7
    DIVIDER_H = 24   # total y advance for divider block
    SEC_HDR_H = 28   # total y advance for section_header()
    BOTTOM    = 16

    fixed_below = DIVIDER_H + SEC_HDR_H + 2 * STAT_H + STAT_GAP + BOTTOM
    available   = frame_height - y - fixed_below
    CARD_H      = max(50, (available - 3 * CARD_GAP) // 4)  # 3 gaps between 4 cards

    metrics_to_draw = [
        ("Pressure", metrics["Blue_pressure_final"]),
        ("Balance",  metrics["Blue_balance_final"]),
        ("Rotation", metrics["Blue_rotation_final"]),
        ("Edging",   metrics["Blue_edging_final"]),
    ]

    def draw_metric_card(cy, label, score):
        col = _tcolor(score)
        # Card background + border
        draw.rounded_rectangle((panel_x, cy, panel_x+panel_w, cy+CARD_H),
                                radius=12, fill=C["card"], outline=C["border"], width=1)
        # Left accent strip
        draw.rounded_rectangle((panel_x, cy, panel_x+3, cy+CARD_H), radius=3, fill=col)

        # Vertical center for text row
        text_y = cy + (CARD_H - 22) // 2

        # Label (left)
        draw.text((LABEL_X, text_y + 4), label, fill=C["text_soft"], font=F["metric_name"])

        # Score (right-anchored before tag)
        display_score = int(round(_clamp_score(score)))
        ss   = f"{display_score}"
        denom = f"/{MAX_BLUE_SCORE}"
        denom_w = draw.textlength(denom, font=F["metric_den"])
        sc_w = draw.textlength(ss, font=F["metric_score"])
        denom_x = SCORE_RIGHT_X - denom_w
        draw.text((denom_x - sc_w - 3, text_y), ss, fill=C["text"], font=F["metric_score"])
        draw.text((denom_x, text_y + 9), denom, fill=C["muted"], font=F["metric_den"])

        # Tag pill (right-anchored)
        tag_text = _tlabel(display_score)
        tag_tw   = draw.textlength(tag_text, font=F["metric_tag"])
        tag_pad  = (TAG_W - tag_tw) / 2
        tag_y    = cy + (CARD_H - TAG_H) // 2
        draw.rounded_rectangle((TAG_X, tag_y, TAG_X+TAG_W, tag_y+TAG_H),
                                radius=10, fill=(20,36,58), outline=col, width=1)
        draw.text((TAG_X + tag_pad, tag_y + 4), tag_text, fill=col, font=F["metric_tag"])

        # Progress bar (near bottom of card)
        bar_bottom_pad = 10
        bar_y  = cy + CARD_H - bar_bottom_pad - 5
        bar_x  = panel_x + 12
        bar_w  = panel_w - 24
        fw     = int(bar_w * _score_fill_ratio(display_score))
        draw.rounded_rectangle((bar_x, bar_y, bar_x+bar_w, bar_y+5), radius=3, fill=C["track"])
        if fw > 0:
            draw.rounded_rectangle((bar_x, bar_y, bar_x+fw, bar_y+5), radius=3, fill=col)

    for i, (label, score) in enumerate(metrics_to_draw):
        draw_metric_card(y, label, score)
        y += CARD_H + (CARD_GAP if i < len(metrics_to_draw)-1 else 0)

    # ── DIVIDER ──────────────────────────────────────────────────────────────
    y += 6
    draw.line((panel_x, y, panel_x+panel_w, y), fill=(38, 50, 72), width=1)
    y += DIVIDER_H - 6

    # ── THIS RUN / LIVE RUN METRICS ──────────────────────────────────────────
    sec_title = "LIVE RUN METRICS" if display_mode == "coach" else "THIS RUN"
    y = section_header(y, sec_title, live=(display_mode == "coach"))

    if display_mode == "coach":
        # Coach: 2×2 live metric grid
        live_items = [
            ("Ski Separation", f"{_latest_metric_value(metrics,'ski_angle')}\u00b0"),
            ("Edge Angle",     f"{_latest_metric_value(metrics,'ski_angle2')}\u00b0"),
            ("Hip Angle",      f"{_latest_metric_value(metrics,'hip_angle')}\u00b0"),
            ("Bend Angle",     f"{_latest_metric_value(metrics,'bend_angle')}\u00b0"),
        ]
        LCOL_W  = (panel_w - 8) // 2
        LCARD_H = STAT_H
        LGAP    = 8
        for idx, (lbl, val) in enumerate(live_items):
            col_idx = idx % 2
            row_idx = idx // 2
            lx = panel_x + col_idx * (LCOL_W + LGAP)
            ly = y + row_idx * (LCARD_H + LGAP)
            if ly + LCARD_H > frame_height - BOTTOM:
                break
            draw.rounded_rectangle((lx, ly, lx+LCOL_W, ly+LCARD_H),
                                    radius=10, fill=C["card_soft"], outline=(38,50,72), width=1)
            draw.text((lx+12, ly+8),  lbl, fill=C["faint"], font=F["stat_label"])
            draw.text((lx+12, ly+26), val, fill=C["blue"],  font=F["stat_value"])
    else:
        # Athlete mode: two full-width stat rows (label left | value right, inline)
        turns_val = _latest_metric_value(metrics, "turns", precision=0)
        speed_raw = _latest_metric_value(metrics, "speed/lateral movement", precision=1)
        speed_str = f"{speed_raw} px/frame" if speed_raw != "N/A" else "N/A"

        stat_items = [
            ("Turns completed", turns_val),
            ("Speed",           speed_str),
        ]

        for idx, (lbl, val) in enumerate(stat_items):
            sy = y + idx * (STAT_H + STAT_GAP)
            if sy + STAT_H > frame_height - BOTTOM:
                break
            # Card background
            draw.rounded_rectangle((panel_x, sy, panel_x+panel_w, sy+STAT_H),
                                    radius=12, fill=C["card_soft"], outline=(38,50,72), width=1)
            # Label — left side, vertically centered
            draw.text((panel_x+16, sy + (STAT_H - 14) // 2), lbl,
                      fill=C["muted"], font=F["stat_label"])
            # Value — right side, vertically centered, bold blue
            vw = draw.textlength(val, font=F["stat_value"])
            draw.text((panel_x + panel_w - 14 - vw, sy + (STAT_H - 16) // 2), val,
                      fill=C["blue"], font=F["stat_value"])

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def create_overlay(frame, metrics, frame_number, TARGET_WIDTH, logo_path=None, display_mode="coach"):
    """Create the polished analysis sidebar without changing scoring behavior."""
    logger.debug(f"Creating overlay for frame {frame_number}")

    try:
        SIDEBAR_WIDTH = 720
        EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
        frame_height = frame.shape[0]
        sidebar_x = TARGET_WIDTH

        colors = {
            "bg": (242, 244, 246),
            "panel": (255, 255, 255),
            "text": (36, 42, 48),
            "muted": (118, 126, 134),
            "line": (213, 219, 224),
            "bar_bg": (228, 233, 237),
            "accent": (190, 102, 12),
            "accent_light": (238, 220, 195),
        }

        extended_frame = np.zeros((frame_height, EXTENDED_WIDTH, 3), dtype=np.uint8)
        extended_frame[:, :TARGET_WIDTH] = frame
        extended_frame[:, TARGET_WIDTH:] = colors["bg"]
        overlay = extended_frame.copy()

        cv2.line(overlay, (TARGET_WIDTH, 0), (TARGET_WIDTH, frame_height), colors["line"], 2)

        logo_w = 500
        logo_h = 86
        logo_x = sidebar_x + (SIDEBAR_WIDTH - logo_w) // 2
        logo_y = 18
        logo_loaded = False

        if logo_path and os.path.exists(logo_path):
            try:
                logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
                if logo_img is not None:
                    logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
                    if len(logo_resized.shape) == 3 and logo_resized.shape[2] == 4:
                        alpha = logo_resized[:, :, 3] / 255.0
                        logo_rgb = logo_resized[:, :, :3]
                        for c in range(3):
                            overlay[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w, c] = (
                                alpha * logo_rgb[:, :, c] +
                                (1 - alpha) * overlay[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w, c]
                            )
                    else:
                        overlay[logo_y:logo_y + logo_h, logo_x:logo_x + logo_w] = logo_resized
                    logo_loaded = True
            except Exception as e:
                logger.error(f"Error loading logo: {e}")

        if not logo_loaded:
            draw_logo_placeholder(overlay, logo_x, logo_y, logo_w, logo_h, colors["line"], colors["text"])

        panel_x = sidebar_x + 28
        panel_w = SIDEBAR_WIDTH - 56
        score_panel_y = 118
        score_panel_h = 430
        cv2.rectangle(overlay, (panel_x, score_panel_y), (panel_x + panel_w, score_panel_y + score_panel_h), colors["panel"], -1)
        cv2.rectangle(overlay, (panel_x, score_panel_y), (panel_x + panel_w, score_panel_y + score_panel_h), colors["line"], 1)

        cv2.putText(overlay, "PERFORMANCE PROFILE", (panel_x + 24, score_panel_y + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, colors["text"], 2)
        cv2.putText(overlay, display_mode.upper(), (panel_x + panel_w - 118, score_panel_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors["muted"], 1)

        blue_iq = (
            metrics["Blue_pressure_final"] +
            metrics["Blue_rotation_final"] +
            metrics["Blue_edging_final"] +
            metrics["Blue_balance_final"]
        ) / 4
        overall_tier, overall_color = _score_tier(blue_iq)

        cv2.rectangle(overlay, (panel_x + 24, score_panel_y + 65),
                      (panel_x + panel_w - 24, score_panel_y + 130), colors["bg"], -1)
        cv2.putText(overlay, "BLUE IQ", (panel_x + 46, score_panel_y + 105),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, colors["accent"], 2)
        cv2.putText(overlay, f"{blue_iq:.0f}", (panel_x + panel_w - 155, score_panel_y + 108),
                    cv2.FONT_HERSHEY_DUPLEX, 1.25, overall_color, 3)
        cv2.putText(overlay, overall_tier.upper(), (panel_x + panel_w - 150, score_panel_y + 127),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors["muted"], 1)

        tier_y = score_panel_y + 162
        tier_labels = [
            ("NEEDS WORK", (74, 96, 214)),
            ("DEVELOPING", (42, 157, 244)),
            ("STRONG", (108, 166, 32)),
            ("EXCELLENT", (42, 42, 42)),
        ]
        tier_w = (panel_w - 48) // 4
        for i, (label, color) in enumerate(tier_labels):
            x1 = panel_x + 24 + i * tier_w
            x2 = panel_x + 24 + (i + 1) * tier_w - 4
            cv2.rectangle(overlay, (x1, tier_y), (x2, tier_y + 24), color, -1)
            cv2.putText(overlay, label, (x1 + 10, tier_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2)

        metrics_to_draw = [
            ("Pressure", metrics["Blue_pressure_final"]),
            ("Balance", metrics["Blue_balance_final"]),
            ("Rotation", metrics["Blue_rotation_final"]),
            ("Edging", metrics["Blue_edging_final"]),
        ]
        row_y = score_panel_y + 204
        for label, score in metrics_to_draw:
            _draw_metric_bar(overlay, label, score, panel_x + 24, row_y, panel_w - 48, colors)
            row_y += 56

        cv2.rectangle(overlay, (panel_x, 570), (panel_x + panel_w, 704), colors["panel"], -1)
        cv2.rectangle(overlay, (panel_x, 570), (panel_x + panel_w, 704), colors["line"], 1)
        cv2.putText(overlay, "LIVE RUN METRICS", (panel_x + 24, 606),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, colors["text"], 2)

        live_metrics = [
            ("Ski Separation", f"{_latest_metric_value(metrics, 'ski_angle')} deg"),
            ("Edge Angle", f"{_latest_metric_value(metrics, 'ski_angle2')} deg"),
            ("Turns", _latest_metric_value(metrics, "turns", precision=0)),
        ]
        metric_y = 638
        for label, value in live_metrics:
            cv2.putText(overlay, label, (panel_x + 32, metric_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colors["text"], 2)
            cv2.putText(overlay, value, (panel_x + panel_w - 170, metric_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, colors["accent"], 2)
            metric_y += 28

        sidebar_region = extended_frame[:, TARGET_WIDTH:]
        overlay_sidebar = overlay[:, TARGET_WIDTH:]
        extended_frame[:, TARGET_WIDTH:] = cv2.addWeighted(overlay_sidebar, 0.98, sidebar_region, 0.02, 0)

        logger.debug(f"Overlay created successfully for frame {frame_number}")
        return extended_frame

    except Exception as e:
        logger.error(f"Error creating overlay for frame {frame_number}: {e}")
        try:
            SIDEBAR_WIDTH = 720
            EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
            frame_height = frame.shape[0]
            extended_frame = np.zeros((frame_height, EXTENDED_WIDTH, 3), dtype=np.uint8)
            extended_frame[:, :TARGET_WIDTH] = frame
            extended_frame[:, TARGET_WIDTH:] = (255, 255, 255)
            return extended_frame
        except Exception:
            logger.critical("Failed to create fallback frame, returning original")
            return frame
