"""Downloadable versions of an analysed run, rendered by background workers.

Every run offers six downloads: Original (16:9 with the score sidebar),
Vertical (9:16) and Square (1:1), each in Athlete (clean footage) or Coach
(technique lines) mode. The Original in the run's own display mode is the
main analysis video; the other five are rendered here.

The analysis loop only *records* what rendering needs (``ExportRecorder``):
the skier's position, the coach-mode line layer stored losslessly, and the
sidebar numbers per analysed frame. After the run is saved, ``submit_exports``
renders each version on a shared worker pool, uploads it, and records it on
the run as soon as it is ready, so the analysis job itself stays fast.

Coach-mode versions can only run at the analysis rate (~10 fps), the same as
the coach overlay video, because the technique lines exist only on analysed
frames. Athlete-mode social crops use every source frame (up to 30 fps).
"""

import json
import logging
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, wait as wait_for_futures
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from services.report_generator import _load_font, _score_level_label
from services.video_encoding import transcode_to_h264

logger = logging.getLogger(__name__)

MODES = ("athlete", "coach")
FORMATS = ("original", "vertical", "square")

# name -> (output width, output height)
SOCIAL_SIZES: Dict[str, Tuple[int, int]] = {
    "vertical": (720, 1280),
    "square": (720, 720),
}

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_SOCIAL_FPS = 30
END_CARD_SECONDS = 2.5
BRAND_BG = (21, 21, 21)          # #151515
BRAND_ACCENT = (139, 192, 213)   # #8BC0D5

# Smoothing applied once per *source* frame so pans look the same whatever
# rate a version is written at.
CENTER_SMOOTHING = 0.12
ZOOM_SMOOTHING = 0.05
# Crop height relative to the skier's height, bounded so distant skiers are
# brought closer without upscaling the footage into blur.
SKIER_HEIGHT_FACTOR = 3.2
MIN_CROP_FRACTION = 0.55

# Sidebar values create_premium_overlay reads besides the four pillar scores.
_SIDEBAR_LATEST_KEYS = (
    "ski_angle",
    "pressure_angle",
    "rotation_angle",
    "balance_angle",
    "turns",
    "speed/lateral movement",
)
_PILLAR_KEYS = (
    "Blue_pressure_final",
    "Blue_balance_final",
    "Blue_rotation_final",
    "Blue_edging_final",
)

# A run still "preparing" after this long was interrupted (e.g. a restart).
PENDING_TIMEOUT = timedelta(hours=2)
# A live render touches its run at least every HEARTBEAT; one that has been
# silent for STALL_AFTER died with its process and is requeued.
HEARTBEAT_SECONDS = 30
STALL_AFTER = timedelta(minutes=2)
MAX_RENDER_ATTEMPTS = 3

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluerun.png")

_pool: Optional[ThreadPoolExecutor] = None
_pool_lock = threading.Lock()
_db_lock = threading.Lock()


def export_workers() -> int:
    try:
        return max(1, int(os.getenv("EXPORT_WORKERS", "3")))
    except ValueError:
        return 3


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=export_workers(), thread_name_prefix="video-export")
        return _pool


def _encoder_threads() -> int:
    return max(1, (os.cpu_count() or 2) // export_workers())


# ── Recording during analysis ──────────────────────────────────────────────


class ExportRecorder:
    """Collects what the export workers need while the analysis runs."""

    def __init__(self, output_path: str, output_fps: float):
        self.base = os.path.splitext(output_path)[0]
        self.output_fps = float(output_fps)
        self.overlay_path = f"{self.base}_coach_layer.avi"
        self.writer: Optional[cv2.VideoWriter] = None
        self.disabled = False
        self.geometry: Optional[Dict[str, int]] = None
        self.processed_indices: List[int] = []
        self.metrics: List[Dict[str, Any]] = []
        self.targets: Dict[int, List[float]] = {}

    def set_geometry(self, source_frame: np.ndarray, new_width: int, new_height: int) -> None:
        if self.geometry is None:
            self.geometry = {
                "source_width": int(source_frame.shape[1]),
                "source_height": int(source_frame.shape[0]),
                "new_width": int(new_width),
                "new_height": int(new_height),
                "x_offset": (TARGET_WIDTH - int(new_width)) // 2,
                "y_offset": (TARGET_HEIGHT - int(new_height)) // 2,
            }

    def record_target(self, frame_index: int, box_center_x: float, box_center_y: float, box_height: float) -> None:
        """Skier position from the 1280x720 analysis frame, stored in source pixels."""
        if self.disabled or self.geometry is None:
            return
        scale = self.geometry["new_width"] / max(1, self.geometry["source_width"])
        self.targets[int(frame_index)] = [
            (box_center_x - self.geometry["x_offset"]) / scale,
            (box_center_y - self.geometry["y_offset"]) / scale,
            box_height / scale,
        ]

    def record_frame(self, frame_index: int, clean_frame: np.ndarray, coach_frame: np.ndarray, metrics: Dict[str, Any]) -> None:
        """Store the coach line layer (losslessly) and sidebar values for one analysed frame."""
        if self.disabled:
            return
        try:
            if self.writer is None:
                self.writer = cv2.VideoWriter(
                    self.overlay_path,
                    cv2.VideoWriter_fourcc(*"FFV1"),
                    self.output_fps,
                    (TARGET_WIDTH, TARGET_HEIGHT),
                )
                if not self.writer.isOpened():
                    raise RuntimeError("could not open lossless coach-layer writer")
            mask = np.any(coach_frame != clean_frame, axis=2)
            layer = np.zeros_like(coach_frame)
            layer[mask] = coach_frame[mask]
            # Pure black marks "no line"; nudge real black line pixels off zero.
            layer[mask & ~np.any(layer, axis=2)] = 1
            self.writer.write(layer)
            snapshot = {key: float(metrics.get(key, 0.0) or 0.0) for key in _PILLAR_KEYS}
            for key in _SIDEBAR_LATEST_KEYS:
                value = metrics.get(key)
                if isinstance(value, list):
                    value = value[-1] if value else None
                snapshot[key] = [value] if isinstance(value, (int, float, str)) else []
            self.metrics.append(snapshot)
            self.processed_indices.append(int(frame_index))
        except Exception as exc:
            logger.error("Export recording disabled after error: %s", exc)
            self.abort()

    def abort(self) -> None:
        self.disabled = True
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
        _remove_quietly(self.overlay_path)

    def finish(
        self,
        source_path: str,
        rotation: int,
        source_fps: float,
        display_mode: str,
        blue_iq: float,
        label: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Close the recording and return the export manifest (or None)."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.disabled or not self.processed_indices or self.geometry is None:
            self.abort()
            return None
        manifest = {
            "base": self.base,
            "source_path": os.path.abspath(source_path),
            "rotation": int(rotation or 0),
            "source_fps": float(source_fps or 30.0),
            "output_fps": self.output_fps,
            "display_mode": display_mode,
            "geometry": self.geometry,
            "overlay_path": self.overlay_path,
            "processed_indices": self.processed_indices,
            "metrics": self.metrics,
            "targets": [[index, *values] for index, values in sorted(self.targets.items())],
            "blue_iq": float(blue_iq),
            "label": label,
        }
        manifest_path = f"{self.base}_export.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        manifest["manifest_path"] = manifest_path
        return manifest


# ── Branding ───────────────────────────────────────────────────────────────


def _load_logo() -> Optional[Image.Image]:
    try:
        return Image.open(LOGO_PATH).convert("RGBA")
    except Exception as exc:
        logger.warning("Export logo could not be loaded: %s", exc)
        return None


def _watermark(width: int, height: int) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    """White logo with a soft shadow, as BGR + alpha arrays and a position.

    Sized from the frame width and kept in the top-left, clear of the
    bottom caption area and right-hand buttons of Reels/TikTok/Stories.
    """
    logo = _load_logo()
    if logo is None:
        return None
    logo_w = int(width * 0.20)
    logo_h = max(1, int(logo.height * logo_w / logo.width))
    alpha = logo.resize((logo_w, logo_h), Image.LANCZOS).getchannel("A")

    pad = max(6, logo_h // 2)
    canvas_size = (logo_w + pad * 2, logo_h + pad * 2)
    shadow = Image.new("L", canvas_size, 0)
    shadow.paste(alpha.point(lambda a: int(a * 0.55)), (pad, pad + max(1, logo_h // 14)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, logo_h // 6)))

    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer.putalpha(shadow)
    white = Image.new("RGBA", (logo_w, logo_h), (255, 255, 255, 0))
    white.putalpha(alpha.point(lambda a: int(a * 0.92)))
    layer.alpha_composite(white, (pad, pad))

    rgba = np.array(layer)
    bgr = rgba[:, :, 2::-1].astype(np.float32)
    weight = rgba[:, :, 3:4].astype(np.float32) / 255.0
    margin_x = int(width * 0.05)
    top = int(height * 0.07) if height > width else int(width * 0.045)
    return bgr, weight, (max(0, margin_x - pad), max(0, top - pad))


def _apply_watermark(frame: np.ndarray, watermark) -> np.ndarray:
    if watermark is None:
        return frame
    bgr, weight, (x, y) = watermark
    h, w = weight.shape[:2]
    region = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (weight * bgr + (1 - weight) * region).astype(np.uint8)
    return frame


def _end_card(size: Tuple[int, int], blue_iq: Optional[float]) -> np.ndarray:
    width, height = size
    card = Image.new("RGB", size, BRAND_BG)
    draw = ImageDraw.Draw(card)
    center_x = width // 2
    y = int(height * (0.32 if height > width else 0.24))

    logo = _load_logo()
    if logo is not None:
        logo_w = int(width * 0.46)
        logo_h = max(1, int(logo.height * logo_w / logo.width))
        scaled = logo.resize((logo_w, logo_h), Image.LANCZOS)
        card.paste(scaled, (center_x - logo_w // 2, y), scaled)
        y += logo_h + int(height * 0.06)

    if blue_iq is not None:
        score = int(np.ceil(blue_iq))

        def centered(text: str, font, fill, top: int) -> int:
            box = draw.textbbox((0, 0), text, font=font)
            draw.text((center_x - (box[2] - box[0]) // 2 - box[0], top - box[1]), text, font=font, fill=fill)
            return top + (box[3] - box[1])

        y = centered("BLUE IQ SCORE", _load_font(int(width * 0.04), bold=True), (140, 140, 140), y) + int(height * 0.02)
        y = centered(str(score), _load_font(int(width * 0.22), bold=True), (255, 255, 255), y) + int(height * 0.03)
        centered(_score_level_label(score).upper(), _load_font(int(width * 0.05), bold=True), BRAND_ACCENT, y)

    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


# ── Frame sources ──────────────────────────────────────────────────────────


def _source_frames(manifest: Dict[str, Any]):
    """Yield (index, upright source frame) exactly as the analysis read them."""
    from services.new_analysis import _disable_capture_auto_rotation
    from services.new_utils import rotate_frame

    capture = cv2.VideoCapture(manifest["source_path"])
    if not capture.isOpened():
        raise RuntimeError("source video could not be opened")
    _disable_capture_auto_rotation(capture)
    rotation = manifest["rotation"]
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if rotation:
                frame = rotate_frame(frame, rotation)
            yield index, frame
            index += 1
    finally:
        capture.release()


def _coach_layers(manifest: Dict[str, Any]):
    capture = cv2.VideoCapture(manifest["overlay_path"])
    if not capture.isOpened():
        raise RuntimeError("coach line layer could not be opened")
    try:
        while True:
            ok, layer = capture.read()
            if not ok:
                break
            yield layer
    finally:
        capture.release()


def _analysed_frames(manifest: Dict[str, Any], mode: str):
    """Yield (index, position, source frame, coach layer or None) for analysed frames."""
    wanted = manifest["processed_indices"]
    layers = _coach_layers(manifest) if mode == "coach" else None
    position = 0
    for index, frame in _source_frames(manifest):
        if position >= len(wanted):
            break
        if index != wanted[position]:
            continue
        layer = next(layers) if layers is not None else None
        yield index, position, frame, layer
        position += 1


def _composite_on_source(source: np.ndarray, layer: np.ndarray, geometry: Dict[str, int]) -> np.ndarray:
    """Scale the 720p coach line layer onto the full-resolution source frame."""
    x0, y0 = geometry["x_offset"], geometry["y_offset"]
    region = layer[y0:y0 + geometry["new_height"], x0:x0 + geometry["new_width"]]
    size = (source.shape[1], source.shape[0])
    mask = np.any(region, axis=2).astype(np.float32)
    # The layer is black outside the lines, i.e. already premultiplied by the
    # mask, so scaling both and compositing leaves no dark fringes.
    lines = cv2.resize(region, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    alpha = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)[:, :, None]
    return np.clip(source.astype(np.float32) * (1 - alpha) + lines, 0, 255).astype(np.uint8)


# ── Rendering ──────────────────────────────────────────────────────────────


class _CropTracker:
    def __init__(self, manifest: Dict[str, Any]):
        self.targets = {int(row[0]): row[1:] for row in manifest["targets"]}
        self.source_w = manifest["geometry"]["source_width"]
        self.source_h = manifest["geometry"]["source_height"]
        self.current = None
        self.smoothed: Optional[List[float]] = None

    def advance(self, index: int) -> Tuple[float, float, float]:
        if index in self.targets:
            self.current = self.targets[index]
        cx, cy, skier_h = self.current or (self.source_w / 2, self.source_h / 2, float(self.source_h))
        goal = [cx, cy, float(np.clip(skier_h * SKIER_HEIGHT_FACTOR, self.source_h * MIN_CROP_FRACTION, self.source_h))]
        if self.smoothed is None:
            self.smoothed = goal
        else:
            self.smoothed = [
                self.smoothed[0] + CENTER_SMOOTHING * (goal[0] - self.smoothed[0]),
                self.smoothed[1] + CENTER_SMOOTHING * (goal[1] - self.smoothed[1]),
                self.smoothed[2] + ZOOM_SMOOTHING * (goal[2] - self.smoothed[2]),
            ]
        return tuple(self.smoothed)

    def crop(self, frame: np.ndarray, center: Tuple[float, float, float], size: Tuple[int, int]) -> np.ndarray:
        aspect = size[0] / size[1]
        crop_h = min(float(self.source_h), center[2])
        crop_w = crop_h * aspect
        if crop_w > self.source_w:
            crop_w = float(self.source_w)
            crop_h = crop_w / aspect
        w, h = max(2, int(crop_w)), max(2, int(crop_h))
        x0 = int(np.clip(center[0] - w / 2, 0, self.source_w - w))
        y0 = int(np.clip(center[1] - h / 2, 0, self.source_h - h))
        crop = frame[y0:y0 + h, x0:x0 + w]
        interpolation = cv2.INTER_AREA if h > size[1] else cv2.INTER_LINEAR
        return cv2.resize(crop, size, interpolation=interpolation)


def _open_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {path}")
    return writer


def _frame_total(manifest: Dict[str, Any], analysed_only: bool) -> int:
    indices = manifest.get("processed_indices") or [0]
    return len(indices) if analysed_only else max(indices) + 1


def _render_social(manifest: Dict[str, Any], mode: str, fmt: str, path: str, progress=None) -> None:
    size = SOCIAL_SIZES[fmt]
    tracker = _CropTracker(manifest)
    watermark = _watermark(*size)
    written = 0

    if mode == "athlete":
        stride = max(1, int(round(manifest["source_fps"] / MAX_SOCIAL_FPS)))
        fps = manifest["source_fps"] / stride
        writer = _open_writer(path, fps, size)
        try:
            total = _frame_total(manifest, analysed_only=False)
            for index, frame in _source_frames(manifest):
                center = tracker.advance(index)
                if index % stride:
                    continue
                writer.write(_apply_watermark(tracker.crop(frame, center, size), watermark))
                written += 1
                if progress:
                    progress(index / total)
            _write_end_card(writer, size, fps, manifest["blue_iq"])
        finally:
            writer.release()
    else:
        fps = manifest["output_fps"]
        writer = _open_writer(path, fps, size)
        geometry = manifest["geometry"]
        last_index = -1
        try:
            for index, _, frame, layer in _analysed_frames(manifest, "coach"):
                # Keep pan smoothing in source-frame time between analysed frames.
                for step in range(last_index + 1, index + 1):
                    center = tracker.advance(step)
                last_index = index
                composite = _composite_on_source(frame, layer, geometry)
                writer.write(_apply_watermark(tracker.crop(composite, center, size), watermark))
                written += 1
                if progress:
                    progress(written / _frame_total(manifest, analysed_only=True))
            _write_end_card(writer, size, fps, manifest["blue_iq"])
        finally:
            writer.release()

    if written == 0:
        raise RuntimeError("no frames rendered")


def _write_end_card(writer: cv2.VideoWriter, size: Tuple[int, int], fps: float, blue_iq: float) -> None:
    card = _end_card(size, blue_iq)
    for _ in range(int(round(fps * END_CARD_SECONDS))):
        writer.write(card)


def _render_original(manifest: Dict[str, Any], mode: str, path: str, progress=None) -> None:
    """The 16:9 sidebar video in the mode the run was *not* analysed in."""
    from services.new_utils import resize_and_center_frame
    from services.overlay_renderers import create_premium_overlay

    label = manifest.get("label") or {}
    writer = None
    written = 0
    try:
        for _, position, frame, layer in _analysed_frames(manifest, mode):
            display, _, _ = resize_and_center_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
            if layer is not None:
                mask = np.any(layer, axis=2)
                display[mask] = layer[mask]
            premium = create_premium_overlay(
                display,
                manifest["metrics"][position],
                position,
                TARGET_WIDTH,
                LOGO_PATH,
                mode,
                athlete_name=label.get("athlete_name"),
                attempt_number=label.get("attempt_number"),
                session_date=label.get("session_date"),
            )
            if writer is None:
                writer = _open_writer(path, manifest["output_fps"], (premium.shape[1], premium.shape[0]))
            writer.write(premium)
            written += 1
            if progress:
                progress(written / _frame_total(manifest, analysed_only=True))
    finally:
        if writer is not None:
            writer.release()
    if written != len(manifest["processed_indices"]):
        raise RuntimeError(f"rendered {written} of {len(manifest['processed_indices'])} frames")


def render_variant(manifest: Dict[str, Any], mode: str, fmt: str, progress=None) -> str:
    """Render one version. ``progress(fraction)`` gets 0..1 updates (optional)."""
    path = f"{manifest['base']}_{mode}_{fmt}.mp4"
    # Frames are ~80% of the work; encoding the H.264 file is the rest.
    frames_progress = (lambda f: progress(0.8 * min(1.0, f))) if progress else None
    try:
        if fmt == "original":
            _render_original(manifest, mode, path, frames_progress)
        else:
            _render_social(manifest, mode, fmt, path, frames_progress)
        if progress:
            progress(0.85)
        transcode_to_h264(path, preset="veryfast", threads=_encoder_threads(), low_priority=True)
        return path
    except Exception:
        _remove_quietly(path)
        raise


def planned_variants(display_mode: str) -> List[Tuple[str, str]]:
    return [
        (mode, fmt)
        for mode in MODES
        for fmt in FORMATS
        if not (fmt == "original" and mode == display_mode)
    ]


# ── Scheduling and bookkeeping ─────────────────────────────────────────────


def _remove_quietly(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _update_run(analysis_id: int, change: Callable[[Any], None]) -> None:
    """Serialised read-modify-write of one run's export fields."""
    from database import SessionLocal, close_session_quietly
    from models.video_analysis import VideoAnalysis

    with _db_lock:
        db = SessionLocal()
        try:
            run = db.query(VideoAnalysis).filter(VideoAnalysis.id == analysis_id).first()
            if run is None:
                return
            change(run)
            run.exports_updated_at = datetime.utcnow()
            db.commit()
        except Exception as exc:
            logger.error("Could not update exports for analysis %s: %s", analysis_id, exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            close_session_quietly(db)


def _store_variant(analysis_id: int, mode: str, fmt: str, path: str) -> None:
    from services.aws_s3 import S3Manager

    entry: Dict[str, Any] = {"path": path}
    if S3Manager.is_enabled():
        key = S3Manager.upload_video(path)
        if key:
            entry = {"s3_key": key}
            _remove_quietly(path)

    def change(run):
        exports = {m: dict(v) for m, v in (run.video_exports or {}).items()}
        exports.setdefault(mode, {})[fmt] = entry
        run.video_exports = exports  # new object so SQLAlchemy sees the change

    _update_run(analysis_id, change)


def _progress_reporter(analysis_id: int, key: str):
    """Write render progress to the run, at most every 10%."""
    last = {"step": -1}

    def report(fraction: float) -> None:
        step = int(max(0.0, min(1.0, fraction)) * 10)
        if step <= last["step"]:
            return
        last["step"] = step

        def change(run):
            values = dict(run.export_progress or {})
            values[key] = step * 10
            run.export_progress = values

        _update_run(analysis_id, change)

    return report


def _run_variant(analysis_id: int, manifest: Dict[str, Any], mode: str, fmt: str) -> bool:
    try:
        path = render_variant(manifest, mode, fmt, _progress_reporter(analysis_id, _variant_key(mode, fmt)))
        _store_variant(analysis_id, mode, fmt, path)
        logger.info("Export ready for analysis %s: %s/%s", analysis_id, mode, fmt)
        return True
    except Exception as exc:
        logger.error("Export failed for analysis %s (%s/%s): %s", analysis_id, mode, fmt, exc)
        return False


SOURCE_PREFIX = "export-sources"


def _variant_key(mode: str, fmt: str) -> str:
    return f"{mode}:{fmt}"


def _parse_variant(value: str) -> Optional[Tuple[str, str]]:
    mode, _, fmt = str(value).partition(":")
    return (mode, fmt) if mode in MODES and fmt in FORMATS else None


def eager_variants(display_mode: str) -> List[Tuple[str, str]]:
    """Versions rendered right after analysis (EXPORTS_EAGER, default athlete:vertical).

    Everything else is rendered the first time someone asks for it.
    """
    planned = planned_variants(display_mode)
    raw = os.getenv("EXPORTS_EAGER", "athlete:vertical").strip().lower()
    if raw == "all":
        return planned
    wanted = {_parse_variant(item.strip()) for item in raw.split(",")}
    return [variant for variant in planned if variant in wanted]


def source_retention_days() -> int:
    try:
        return max(1, int(os.getenv("EXPORT_SOURCE_RETENTION_DAYS", "30")))
    except ValueError:
        return 30


def _portable_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The manifest without machine-specific paths."""
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"source_path", "overlay_path", "base", "manifest_path"}
    }


def retain_source(analysis_id: int, manifest: Dict[str, Any], source_s3_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Keep the footage needed to render more versions later.

    Stored in S3 (under export-sources/, expired by a bucket lifecycle rule)
    when S3 is enabled; otherwise left on local disk (development).
    """
    from services.aws_s3 import S3Manager

    expires_at = (datetime.utcnow() + timedelta(days=source_retention_days())).isoformat()
    portable = _portable_manifest(manifest)
    if not S3Manager.is_enabled():
        return {
            "storage": "local",
            "source_path": manifest["source_path"],
            "layer_path": manifest["overlay_path"],
            "manifest": portable,
            "expires_at": expires_at,
        }

    prefix = f"{SOURCE_PREFIX}/{analysis_id}"
    extension = os.path.splitext(manifest["source_path"])[1] or ".mp4"
    source_key = source_s3_key or S3Manager.upload_file(manifest["source_path"], f"{prefix}/source{extension}", "video/mp4")
    layer_key = S3Manager.upload_file(manifest["overlay_path"], f"{prefix}/coach_layer.avi", "video/x-msvideo")
    manifest_file = f"{manifest['base']}_portable.json"
    try:
        with open(manifest_file, "w", encoding="utf-8") as handle:
            json.dump(portable, handle)
        manifest_key = S3Manager.upload_file(manifest_file, f"{prefix}/manifest.json", "application/json")
    finally:
        _remove_quietly(manifest_file)
    if not (source_key and layer_key and manifest_key):
        logger.error("Could not retain export source for analysis %s; extra versions will be unavailable", analysis_id)
        return None
    return {
        "storage": "s3",
        "source_key": source_key,
        "layer_key": layer_key,
        "manifest_key": manifest_key,
        "expires_at": expires_at,
    }


def source_entries(run: Any, field: str) -> List[str]:
    """Retained-source S3 keys (field="s3_key") or local paths (field="path")."""
    source = getattr(run, "export_source", None) or {}
    if field == "s3_key":
        names = ("source_key", "layer_key", "manifest_key")
    else:
        names = ("source_path", "layer_path")
    return [source[name] for name in names if source.get(name)]


def source_available(source: Optional[Dict[str, Any]]) -> bool:
    if not source:
        return False
    try:
        return datetime.fromisoformat(source["expires_at"]) > datetime.utcnow()
    except (KeyError, ValueError, TypeError):
        return False


def _remove_path(path: Optional[str]) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        _remove_quietly(path)


def _touch(analysis_id: int) -> None:
    _update_run(analysis_id, lambda run: None)  # _update_run stamps exports_updated_at


def _finalize(analysis_id: int, variants, futures, cleanup_paths: List[str]) -> None:
    # Heartbeat until every render finishes, so the watchdog knows they're alive.
    pending = set(futures)
    while pending:
        _, pending = wait_for_futures(pending, timeout=HEARTBEAT_SECONDS)
        if pending:
            _touch(analysis_id)
    results = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception:
            results.append(False)
    finished = {_variant_key(mode, fmt) for mode, fmt in variants}
    outcomes = {_variant_key(mode, fmt): ok for (mode, fmt), ok in zip(variants, results)}
    remaining: List[str] = []
    to_notify: Dict[str, List[int]] = {}

    def change(run):
        # Requests that arrived while rendering stay queued for the next round.
        remaining.extend(r for r in (run.export_requests or []) if r not in finished)
        run.export_requests = remaining or None
        requesters = dict(run.export_requesters or {})
        for key in finished:
            if requesters.get(key):
                to_notify[key] = list(requesters.pop(key))
        run.export_requesters = requesters or None
        progress_values = {k: v for k, v in (run.export_progress or {}).items() if k not in finished}
        run.export_progress = progress_values or None
        run.export_attempts = None
        if remaining:
            run.exports_status = "requested"
        else:
            run.exports_status = "ready" if any(results) or not results else "failed"

    _update_run(analysis_id, change)
    if to_notify:
        from services.notifications import notify_version_result

        for key, people in to_notify.items():
            mode, fmt = _parse_variant(key) or (None, None)
            if mode:
                notify_version_result(analysis_id, mode, fmt, people, bool(outcomes.get(key)))
    for path in cleanup_paths:
        _remove_path(path)
    logger.info("Exports finished for analysis %s (%s of %s ready)", analysis_id, sum(results), len(results))
    if remaining:
        from services.job_queue import RUNNER_INLINE, runner_mode

        if runner_mode() == RUNNER_INLINE:
            process_export_requests(analysis_id)


def _launch(analysis_id: int, manifest: Dict[str, Any], variants, cleanup_paths: List[str]) -> None:
    pool = _get_pool()
    futures = [pool.submit(_run_variant, analysis_id, manifest, mode, fmt) for mode, fmt in variants]
    threading.Thread(
        target=_finalize,
        args=(analysis_id, list(variants), futures, list(cleanup_paths)),
        name=f"video-export-finalize-{analysis_id}",
        daemon=True,
    ).start()


def start_exports(
    analysis_id: int,
    manifest: Dict[str, Any],
    cleanup_paths: Optional[List[str]] = None,
    source_s3_key: Optional[str] = None,
) -> None:
    """After analysis: retain the source, render the eager versions now.

    ``cleanup_paths`` (e.g. the local input video) are deleted when done,
    unless the source had to be kept on local disk.
    """
    source = retain_source(analysis_id, manifest, source_s3_key)
    eager = eager_variants(manifest["display_mode"])

    def change(run):
        run.export_source = source
        run.export_requests = [_variant_key(m, f) for m, f in eager] or None
        run.exports_status = "processing" if eager else "ready"

    _update_run(analysis_id, change)
    keep_local = bool(source and source.get("storage") == "local")
    cleanup = [] if keep_local else [manifest["overlay_path"], *(cleanup_paths or [])]
    cleanup.append(manifest.get("manifest_path"))
    if eager:
        _launch(analysis_id, manifest, eager, cleanup)
    else:
        for path in cleanup:
            _remove_path(path)


def submit_exports(analysis_id: int, manifest: Dict[str, Any], cleanup_paths: Optional[List[str]] = None) -> None:
    """Render every missing version now (used when on-demand is not wanted)."""

    def change(run):
        run.export_requests = [_variant_key(m, f) for m, f in planned_variants(manifest["display_mode"])]
        run.exports_status = "processing"

    _update_run(analysis_id, change)
    _launch(
        analysis_id,
        manifest,
        planned_variants(manifest["display_mode"]),
        [manifest["overlay_path"], manifest.get("manifest_path"), *(cleanup_paths or [])],
    )


def request_variant(analysis_id: int, mode: str, fmt: str, requester_id: Optional[int] = None) -> None:
    """Ask for one version; the worker (or this process when inline) renders it."""
    key = _variant_key(mode, fmt)

    def change(run):
        requests = list(run.export_requests or [])
        if key not in requests:
            requests.append(key)
        run.export_requests = requests
        if requester_id is not None:
            requesters = {k: list(v) for k, v in (run.export_requesters or {}).items()}
            people = requesters.setdefault(key, [])
            if requester_id not in people:
                people.append(requester_id)
            run.export_requesters = requesters
        if run.exports_status != "processing":
            run.exports_status = "requested"

    _update_run(analysis_id, change)
    from services.job_queue import RUNNER_INLINE, runner_mode

    if runner_mode() == RUNNER_INLINE:
        threading.Thread(
            target=process_export_requests,
            args=(analysis_id,),
            name=f"video-export-request-{analysis_id}",
            daemon=True,
        ).start()


def _claim_requests(analysis_id: int) -> bool:
    from sqlalchemy import text as sql_text

    from database import SessionLocal, close_session_quietly

    db = SessionLocal()
    try:
        claimed = db.execute(
            sql_text(
                "UPDATE video_analysis SET exports_status = 'processing', exports_updated_at = :now "
                "WHERE id = :id AND exports_status = 'requested'"
            ),
            {"id": analysis_id, "now": datetime.utcnow()},
        ).rowcount
        db.commit()
        return claimed == 1
    finally:
        close_session_quietly(db)


def _materialise_source(analysis_id: int, source: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return a runnable manifest plus the temporary paths to delete afterwards."""
    if source.get("storage") == "local":
        manifest = dict(source["manifest"])
        manifest.update({
            "source_path": source["source_path"],
            "overlay_path": source["layer_path"],
            "base": os.path.splitext(source["source_path"])[0] + f"_ondemand_{analysis_id}",
        })
        if not (os.path.exists(manifest["source_path"]) and os.path.exists(manifest["overlay_path"])):
            raise RuntimeError("retained source files are missing")
        return manifest, []

    from services.aws_s3 import AWS_S3_BUCKET, S3Manager

    work_dir = os.path.abspath(os.path.join("temp_videos", f"export_{analysis_id}_{uuid.uuid4().hex[:8]}"))
    os.makedirs(work_dir, exist_ok=True)
    try:
        paths = {
            "manifest": os.path.join(work_dir, "manifest.json"),
            "source": os.path.join(work_dir, "source" + (os.path.splitext(source["source_key"])[1] or ".mp4")),
            "layer": os.path.join(work_dir, "coach_layer.avi"),
        }
        for name, key in (("manifest", "manifest_key"), ("source", "source_key"), ("layer", "layer_key")):
            if not S3Manager.download_file(AWS_S3_BUCKET, source[key], paths[name]):
                raise RuntimeError(f"could not download {name} from S3")
        with open(paths["manifest"], encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update({
            "source_path": paths["source"],
            "overlay_path": paths["layer"],
            "base": os.path.join(work_dir, f"run_{analysis_id}"),
        })
        return manifest, [work_dir]
    except Exception:
        _remove_path(work_dir)
        raise


def process_export_requests(analysis_id: int) -> bool:
    """Render a run's requested versions. Safe to call from several places."""
    if not _claim_requests(analysis_id):
        return False

    from database import SessionLocal, close_session_quietly
    from models.video_analysis import VideoAnalysis

    db = SessionLocal()
    try:
        run = db.query(VideoAnalysis).filter(VideoAnalysis.id == analysis_id).first()
        source = dict(run.export_source or {}) if run else {}
        ready = run.available_downloads if run else {}
        wanted = []
        for value in (run.export_requests or []) if run else []:
            variant = _parse_variant(value)
            if variant and variant[1] not in ready.get(variant[0], []) and variant not in wanted:
                wanted.append(variant)
    finally:
        close_session_quietly(db)

    try:
        if not wanted or not source_available(source):
            raise RuntimeError("nothing to render or source expired")
        manifest, temp_paths = _materialise_source(analysis_id, source)
    except Exception as exc:
        logger.warning("On-demand export for analysis %s skipped: %s", analysis_id, exc)

        def clear(run):
            run.export_requests = None
            run.exports_status = "ready"

        _update_run(analysis_id, clear)
        return False

    _launch(analysis_id, manifest, wanted, temp_paths)
    return True


def recover_stalled_exports() -> int:
    """Requeue renders whose process died (no heartbeat for STALL_AFTER).

    After MAX_RENDER_ATTEMPTS the versions are given up on and whoever asked
    is told they could not be made. Returns how many runs were handled.
    """
    from database import SessionLocal, close_session_quietly
    from models.video_analysis import VideoAnalysis

    cutoff = datetime.utcnow() - STALL_AFTER
    db = SessionLocal()
    try:
        stalled = [
            row.id
            for row in db.query(VideoAnalysis.id).filter(
                VideoAnalysis.exports_status == "processing",
                VideoAnalysis.exports_updated_at < cutoff,
            )
        ]
    finally:
        close_session_quietly(db)

    handled = 0
    for analysis_id in stalled:
        given_up: Dict[str, List[int]] = {}
        retried = {"value": False}

        def change(run):
            # Re-check under the lock: a heartbeat may have arrived meanwhile.
            if run.exports_status != "processing" or (run.exports_updated_at and run.exports_updated_at >= cutoff):
                return
            attempts = (run.export_attempts or 0) + 1
            run.export_progress = None
            if attempts < MAX_RENDER_ATTEMPTS and run.export_requests:
                run.export_attempts = attempts
                run.exports_status = "requested"
                retried["value"] = True
            else:
                given_up.update({k: list(v) for k, v in (run.export_requesters or {}).items()})
                run.export_requests = None
                run.export_requesters = None
                run.export_attempts = None
                run.exports_status = "failed"

        _update_run(analysis_id, change)
        if retried["value"]:
            handled += 1
            logger.warning("Requeued stalled download versions for analysis %s", analysis_id)
            from services.job_queue import RUNNER_INLINE, runner_mode

            if runner_mode() == RUNNER_INLINE:
                threading.Thread(
                    target=process_export_requests,
                    args=(analysis_id,),
                    name=f"video-export-request-{analysis_id}",
                    daemon=True,
                ).start()
        elif given_up:
            handled += 1
            logger.error("Gave up on download versions for analysis %s after repeated stalls", analysis_id)
            from services.notifications import notify_version_result

            for key, people in given_up.items():
                variant = _parse_variant(key)
                if variant:
                    notify_version_result(analysis_id, variant[0], variant[1], people, ok=False)
    return handled


_watchdog_started = False


def start_export_watchdog(interval_seconds: int = 60) -> None:
    """Background check for stalled renders (API process in inline mode)."""
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True

    def loop():
        import time

        time.sleep(5)  # let start-up finish
        while True:
            try:
                recover_stalled_exports()
            except Exception as exc:
                logger.warning("Export watchdog check failed: %s", exc)
            time.sleep(interval_seconds)

    threading.Thread(target=loop, name="export-watchdog", daemon=True).start()


def claim_pending_requests(limit: int = 3) -> int:
    """Worker hook: start rendering runs whose versions were requested."""
    from database import SessionLocal, close_session_quietly
    from models.video_analysis import VideoAnalysis

    db = SessionLocal()
    try:
        ids = [
            row.id
            for row in db.query(VideoAnalysis.id)
            .filter(VideoAnalysis.exports_status == "requested")
            .order_by(VideoAnalysis.exports_updated_at.asc().nullsfirst())
            .limit(limit)
        ]
    finally:
        close_session_quietly(db)
    return sum(1 for analysis_id in ids if _source_reachable(analysis_id) and process_export_requests(analysis_id))


def _source_reachable(analysis_id: int) -> bool:
    """Skip runs whose retained footage is on another machine's disk."""
    from database import SessionLocal, close_session_quietly
    from models.video_analysis import VideoAnalysis

    db = SessionLocal()
    try:
        run = db.query(VideoAnalysis).filter(VideoAnalysis.id == analysis_id).first()
        source = (run.export_source or {}) if run else {}
    finally:
        close_session_quietly(db)
    if source.get("storage") == "local":
        return os.path.exists(source.get("source_path") or "")
    return True


def export_entries(run: Any, field: str) -> List[str]:
    """Every stored ``s3_key`` or ``path`` of a run's extra versions."""
    values = []
    for formats in (getattr(run, "video_exports", None) or {}).values():
        for entry in (formats or {}).values():
            if isinstance(entry, dict) and entry.get(field):
                values.append(entry[field])
    return values


def is_pending(status: Optional[str], updated_at: Optional[datetime]) -> bool:
    if status not in {"processing", "requested"}:
        return False
    return updated_at is None or datetime.utcnow() - updated_at < PENDING_TIMEOUT

