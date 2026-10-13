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
import threading
from concurrent.futures import ThreadPoolExecutor
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


def _render_social(manifest: Dict[str, Any], mode: str, fmt: str, path: str) -> None:
    size = SOCIAL_SIZES[fmt]
    tracker = _CropTracker(manifest)
    watermark = _watermark(*size)
    written = 0

    if mode == "athlete":
        stride = max(1, int(round(manifest["source_fps"] / MAX_SOCIAL_FPS)))
        fps = manifest["source_fps"] / stride
        writer = _open_writer(path, fps, size)
        try:
            for index, frame in _source_frames(manifest):
                center = tracker.advance(index)
                if index % stride:
                    continue
                writer.write(_apply_watermark(tracker.crop(frame, center, size), watermark))
                written += 1
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
            _write_end_card(writer, size, fps, manifest["blue_iq"])
        finally:
            writer.release()

    if written == 0:
        raise RuntimeError("no frames rendered")


def _write_end_card(writer: cv2.VideoWriter, size: Tuple[int, int], fps: float, blue_iq: float) -> None:
    card = _end_card(size, blue_iq)
    for _ in range(int(round(fps * END_CARD_SECONDS))):
        writer.write(card)


def _render_original(manifest: Dict[str, Any], mode: str, path: str) -> None:
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
    finally:
        if writer is not None:
            writer.release()
    if written != len(manifest["processed_indices"]):
        raise RuntimeError(f"rendered {written} of {len(manifest['processed_indices'])} frames")


def render_variant(manifest: Dict[str, Any], mode: str, fmt: str) -> str:
    path = f"{manifest['base']}_{mode}_{fmt}.mp4"
    try:
        if fmt == "original":
            _render_original(manifest, mode, path)
        else:
            _render_social(manifest, mode, fmt, path)
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


def _run_variant(analysis_id: int, manifest: Dict[str, Any], mode: str, fmt: str) -> bool:
    try:
        path = render_variant(manifest, mode, fmt)
        _store_variant(analysis_id, mode, fmt, path)
        logger.info("Export ready for analysis %s: %s/%s", analysis_id, mode, fmt)
        return True
    except Exception as exc:
        logger.error("Export failed for analysis %s (%s/%s): %s", analysis_id, mode, fmt, exc)
        return False


def _finalize(analysis_id: int, manifest: Dict[str, Any], futures, cleanup_paths: List[str]) -> None:
    results = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception:
            results.append(False)

    def change(run):
        run.exports_status = "ready" if any(results) else "failed"

    _update_run(analysis_id, change)
    for path in [manifest["overlay_path"], manifest.get("manifest_path"), *cleanup_paths]:
        _remove_quietly(path)
    logger.info("Exports finished for analysis %s (%s of %s ready)", analysis_id, sum(results), len(results))


def submit_exports(analysis_id: int, manifest: Dict[str, Any], cleanup_paths: Optional[List[str]] = None) -> None:
    """Queue every missing version on the worker pool; returns immediately.

    ``cleanup_paths`` (e.g. the uploaded input video) are deleted once all
    versions are finished, because rendering re-reads the source footage.
    """
    _update_run(analysis_id, lambda run: setattr(run, "exports_status", "processing"))
    pool = _get_pool()
    futures = [
        pool.submit(_run_variant, analysis_id, manifest, mode, fmt)
        for mode, fmt in planned_variants(manifest["display_mode"])
    ]
    threading.Thread(
        target=_finalize,
        args=(analysis_id, manifest, futures, list(cleanup_paths or [])),
        name=f"video-export-finalize-{analysis_id}",
        daemon=True,
    ).start()


def export_entries(run: Any, field: str) -> List[str]:
    """Every stored ``s3_key`` or ``path`` of a run's extra versions."""
    values = []
    for formats in (getattr(run, "video_exports", None) or {}).values():
        for entry in (formats or {}).values():
            if isinstance(entry, dict) and entry.get(field):
                values.append(entry[field])
    return values


def is_pending(status: Optional[str], updated_at: Optional[datetime]) -> bool:
    if status != "processing":
        return False
    return updated_at is None or datetime.utcnow() - updated_at < PENDING_TIMEOUT

