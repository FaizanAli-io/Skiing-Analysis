import json
import logging
import os

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

logger = logging.getLogger(__name__)
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
    if score < 100:
        return "Needs Work", (74, 96, 214)
    if score < 150:
        return "Developing", (42, 157, 244)
    if score < 200:
        return "Strong", (108, 166, 32)
    return "Excellent", (42, 42, 42)


def _clamp_score(score, min_score=60, max_score=240):
    return max(min_score, min(max_score, float(score or 0)))


def _draw_metric_bar(canvas, label, score, x, y, width, colors, highlight=False):
    score = _clamp_score(score)
    tier_label, tier_color = _score_tier(score)
    bar_x = x + 168
    bar_y = y + 16
    bar_w = width - 265
    bar_h = 18
    score_ratio = (score - 60) / 180
    fill_w = int(bar_w * score_ratio)

    label_color = colors["accent"] if highlight else colors["text"]
    cv2.putText(canvas, label, (x, y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.68, label_color, 2)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), colors["bar_bg"], -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), tier_color, -1)

    for boundary in (100, 150, 200):
        marker_x = bar_x + int(bar_w * ((boundary - 60) / 180))
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
    levels = [
        (60, 80, "Beginner - Level 1"),
        (81, 100, "Beginner - Level 2"),
        (101, 120, "Intermediate - Level 3"),
        (121, 140, "Intermediate - Level 4"),
        (141, 160, "Intermediate - Level 5"),
        (161, 180, "Intermediate - Level 6"),
        (181, 200, "Intermediate - Level 7"),
        (201, 220, "Expert - Level 8"),
        (221, 240, "Expert - Level 9"),
    ]
    score = _clamp_score(score)
    for low, high, label in levels:
        if low <= score <= high:
            return label
    return "Expert - Level 9"


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
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eef3f6;
      --panel: #ffffff;
      --text: #202830;
      --muted: #7a838d;
      --line: #d7dee4;
      --blue: #2799d6;
      --needs: #d6604a;
      --developing: #f49d2a;
      --strong: #4faf79;
      --excellent: #282828;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #111;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
    }}
    .shell {{
      width: min(1480px, 100vw);
      background: var(--bg);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, .35);
    }}
    video {{
      width: 100%;
      height: 100%;
      display: block;
      background: #000;
      object-fit: contain;
    }}
    aside {{
      min-height: 100%;
      padding: 24px;
      border-left: 1px solid var(--line);
    }}
    .brand {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 22px;
    }}
    .logo {{
      font-size: 34px;
      font-weight: 800;
      letter-spacing: 0;
      color: #8ecae2;
    }}
    .badge {{
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      color: {("#aab0cc" if display_mode == "coach" else "#1a5fa0")};
      background: {("#1e1e2e" if display_mode == "coach" else "#e8f4ff")};
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px;
      margin-bottom: 16px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin-bottom: 14px;
    }}
    .hero-score {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px;
      border-radius: 16px;
      background: linear-gradient(135deg, #f8fbfd, #edf4f8);
      margin-bottom: 18px;
    }}
    .hero-score b {{
      display: block;
      color: var(--blue);
      font-size: 18px;
      margin-bottom: 4px;
    }}
    .hero-score span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .score-number {{
      font-size: 48px;
      line-height: 1;
      font-weight: 900;
      color: var(--score-color);
    }}
    .metric {{
      margin-top: 16px;
    }}
    .metric-top {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .metric-name {{
      font-size: 15px;
      font-weight: 800;
    }}
    .metric-value {{
      font-size: 15px;
      font-weight: 900;
      color: var(--metric-color);
    }}
    .bar {{
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: #e5ebef;
      position: relative;
    }}
    .fill {{
      width: var(--pct);
      height: 100%;
      border-radius: 999px;
      background: var(--metric-color);
    }}
    .tier {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .tiers {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 4px;
      margin-bottom: 12px;
    }}
    .tiers div {{
      color: white;
      border-radius: 7px;
      padding: 7px 6px;
      font-size: 10px;
      font-weight: 900;
      text-align: center;
    }}
    .rows {{
      display: grid;
      gap: 12px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      font-size: 14px;
      font-weight: 750;
    }}
    .row span:last-child {{
      color: var(--blue);
      font-weight: 900;
    }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    const data = {json.dumps(payload)};
    const e = React.createElement;
    const tierColor = score => score < 100 ? "var(--needs)" : score < 150 ? "var(--developing)" : score < 200 ? "var(--strong)" : "var(--excellent)";
    const tierLabel = score => score < 100 ? "Needs Work" : score < 150 ? "Developing" : score < 200 ? "Strong" : "Excellent";
    const pct = score => Math.max(0, Math.min(100, ((score - 60) / 180) * 100)) + "%";
    function Metric({{ name, score }}) {{
      const color = tierColor(score);
      return e("div", {{ className: "metric", style: {{ "--metric-color": color }} }},
        e("div", {{ className: "metric-top" }},
          e("div", {{ className: "metric-name" }}, name),
          e("div", {{ className: "metric-value" }}, Math.round(score))
        ),
        e("div", {{ className: "bar" }}, e("div", {{ className: "fill", style: {{ "--pct": pct(score) }} }})),
        e("div", {{ className: "tier" }}, tierLabel(score))
      );
    }}
    function App() {{
      const coach = data.display_mode === "coach";
      const rows = coach
        ? [["Turns", data.turns], ["Duration", data.duration.toFixed(1) + " sec"], ["Processed frames", data.processed_frames]]
        : [["Turns completed", data.turns], ["Duration", data.duration.toFixed(1) + " sec"]];
      return e("main", {{ className: "shell" }},
        e("video", {{ src: data.video_src, controls: true, autoPlay: false }}),
        e("aside", null,
          e("div", {{ className: "brand" }}, e("div", {{ className: "logo" }}, "bluerun"), e("div", {{ className: "badge" }}, data.display_mode.toUpperCase())),
          e("section", {{ className: "card" }},
            e("div", {{ className: "eyebrow" }}, "Performance Profile"),
            e("div", {{ className: "hero-score", style: {{ "--score-color": tierColor(data.blue_iq) }} }},
              e("div", null, e("b", null, "BLUE IQ"), e("span", null, data.level_label)),
              e("div", {{ className: "score-number" }}, Math.round(data.blue_iq))
            ),
            e("div", {{ className: "tiers" }},
              e("div", {{ style: {{ background: "var(--needs)" }} }}, "NEEDS WORK"),
              e("div", {{ style: {{ background: "var(--developing)" }} }}, "DEVELOPING"),
              e("div", {{ style: {{ background: "var(--strong)" }} }}, "STRONG"),
              e("div", {{ style: {{ background: "var(--excellent)" }} }}, "EXCELLENT")
            ),
            e(Metric, {{ name: "Pressure", score: data.pressure_score }}),
            e(Metric, {{ name: "Balance", score: data.balance_score }}),
            e(Metric, {{ name: "Rotation", score: data.rotation_score }}),
            e(Metric, {{ name: "Edging", score: data.edging_score }})
          ),
          e("section", {{ className: "card" }},
            e("div", {{ className: "eyebrow" }}, coach ? "Live Run Metrics" : "This Run"),
            e("div", {{ className: "rows" }}, rows.map(row => e("div", {{ className: "row", key: row[0] }}, e("span", null, row[0]), e("span", null, row[1]))))
          )
        )
      );
    }}
    ReactDOM.createRoot(document.getElementById("root")).render(e(App));
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

    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


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


def create_premium_overlay(frame, metrics, frame_number, TARGET_WIDTH, logo_path=None, display_mode="coach"):
    """Render a higher-quality burned-in sidebar using Pillow."""
    if Image is None:
        raise RuntimeError("Pillow is required for premium overlay rendering")

    SIDEBAR_WIDTH = 720
    frame_height = frame.shape[0]
    video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_img = Image.fromarray(video_rgb)

    canvas = Image.new("RGB", (TARGET_WIDTH + SIDEBAR_WIDTH, frame_height), (242, 244, 246))
    canvas.paste(video_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    colors = {
        "bg": (242, 244, 246),
        "panel": (255, 255, 255),
        "text": (32, 40, 48),
        "muted": (117, 127, 136),
        "muted_dark": (80, 89, 98),
        "line": (213, 219, 224),
        "bar_bg": (228, 233, 237),
        "blue": (39, 153, 214),
        "coach_bg": (30, 30, 46),
        "coach_text": (188, 194, 222),
        "athlete_bg": (232, 244, 255),
        "athlete_text": (26, 95, 160),
        "needs": (214, 96, 74),
        "developing": (244, 157, 42),
        "strong": (79, 175, 121),
        "excellent": (42, 42, 42),
    }
    fonts = {
        "brand": _load_font(46, bold=True),
        "title": _load_font(17, bold=True),
        "badge": _load_font(13, bold=True),
        "hero": _load_font(24, bold=True),
        "hero_score": _load_font(54, bold=True),
        "small": _load_font(14, bold=True),
        "metric": _load_font(18, bold=True),
        "tiny": _load_font(11, bold=True),
        "row": _load_font(16, bold=True),
    }

    sidebar_x = TARGET_WIDTH
    draw.line((sidebar_x, 0, sidebar_x, frame_height), fill=colors["line"], width=1)

    logo_w, logo_h = 420, 72
    logo_x = sidebar_x + (SIDEBAR_WIDTH - logo_w) // 2
    logo_y = 20
    logo_loaded = False
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA").resize((logo_w, logo_h))
            canvas.paste(logo, (logo_x, logo_y), logo)
            logo_loaded = True
        except Exception as e:
            logger.error(f"Premium logo load error: {e}")
    if not logo_loaded:
        draw.text((logo_x + 10, logo_y + 10), "bluerun", fill=(142, 202, 226), font=fonts["brand"])

    badge_label = "COACH" if display_mode == "coach" else "ATHLETE"
    badge_fill = colors["coach_bg"] if display_mode == "coach" else colors["athlete_bg"]
    badge_text = colors["coach_text"] if display_mode == "coach" else colors["athlete_text"]
    badge_x = sidebar_x + SIDEBAR_WIDTH - 146
    badge_y = 38
    draw.rounded_rectangle((badge_x, badge_y, badge_x + 116, badge_y + 31), radius=16, fill=badge_fill)
    draw.text((badge_x + 18, badge_y + 8), badge_label, fill=badge_text, font=fonts["badge"])

    panel_x = sidebar_x + 28
    panel_w = SIDEBAR_WIDTH - 56
    panel_y = 108
    panel_h = 450
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=18, fill=colors["panel"], outline=colors["line"], width=1)
    draw.text((panel_x + 24, panel_y + 24), "PERFORMANCE PROFILE", fill=colors["muted"], font=fonts["title"])

    blue_iq = (
        metrics["Blue_pressure_final"] +
        metrics["Blue_rotation_final"] +
        metrics["Blue_edging_final"] +
        metrics["Blue_balance_final"]
    ) / 4
    overall_tier, overall_color_bgr = _score_tier(blue_iq)
    overall_color = _rgb(overall_color_bgr)

    chip_y = panel_y + 58
    draw.rounded_rectangle((panel_x + 20, chip_y, panel_x + panel_w - 20, chip_y + 78), radius=16, fill=(248, 251, 253))
    draw.text((panel_x + 42, chip_y + 17), "BLUE IQ", fill=colors["blue"], font=fonts["hero"])
    draw.text((panel_x + 42, chip_y + 48), _blueiq_level_label(blue_iq), fill=colors["muted_dark"], font=fonts["small"])
    draw.text((panel_x + panel_w - 166, chip_y + 10), f"{blue_iq:.0f}", fill=overall_color, font=fonts["hero_score"])
    draw.text((panel_x + panel_w - 158, chip_y + 58), overall_tier.upper(), fill=colors["muted_dark"], font=fonts["tiny"])

    tier_y = chip_y + 96
    tier_items = [
        ("NEEDS WORK", colors["needs"]),
        ("DEVELOPING", colors["developing"]),
        ("STRONG", colors["strong"]),
        ("EXCELLENT", colors["excellent"]),
    ]
    tier_w = (panel_w - 48) // 4
    for idx, (label, fill) in enumerate(tier_items):
        x1 = panel_x + 24 + idx * tier_w
        x2 = panel_x + 24 + (idx + 1) * tier_w - 6
        draw.rounded_rectangle((x1, tier_y, x2, tier_y + 27), radius=7, fill=fill)
        draw.text((x1 + 9, tier_y + 8), label, fill=(255, 255, 255), font=fonts["tiny"])

    row_y = tier_y + 44
    for label, score in [
        ("Pressure", metrics["Blue_pressure_final"]),
        ("Balance", metrics["Blue_balance_final"]),
        ("Rotation", metrics["Blue_rotation_final"]),
        ("Edging", metrics["Blue_edging_final"]),
    ]:
        _premium_metric_bar(draw, label, score, panel_x + 24, row_y, panel_w - 48, colors, fonts)
        row_y += 58

    live_y = panel_y + panel_h + 14
    live_h = 154 if display_mode == "coach" else 94
    draw.rounded_rectangle((panel_x, live_y, panel_x + panel_w, live_y + live_h), radius=18, fill=colors["panel"], outline=colors["line"], width=1)
    section_title = "LIVE RUN METRICS" if display_mode == "coach" else "THIS RUN"
    draw.text((panel_x + 24, live_y + 22), section_title, fill=colors["muted"], font=fonts["title"])

    if display_mode == "coach":
        live_rows = [
            ("Ski separation", f"{_latest_metric_value(metrics, 'ski_angle')} deg"),
            ("Edge angle", f"{_latest_metric_value(metrics, 'ski_angle2')} deg"),
            ("Hip angle", f"{_latest_metric_value(metrics, 'hip_angle')} deg"),
            ("Bend angle", f"{_latest_metric_value(metrics, 'bend_angle')} deg"),
            ("Turns", _latest_metric_value(metrics, "turns", precision=0)),
        ]
    else:
        live_rows = [
            ("Turns completed", _latest_metric_value(metrics, "turns", precision=0)),
        ]

    metric_y = live_y + 57
    for label, value in live_rows:
        draw.text((panel_x + 32, metric_y), label, fill=colors["text"], font=fonts["row"])
        draw.text((panel_x + panel_w - 170, metric_y), str(value), fill=colors["blue"], font=fonts["row"])
        metric_y += 24

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




