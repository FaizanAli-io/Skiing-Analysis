"""Shared ffmpeg helpers for producing browser- and social-playable MP4s."""

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def find_ffmpeg() -> Optional[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    for candidate in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def transcode_to_h264(
    path: str,
    preset: str = "medium",
    threads: Optional[int] = None,
    low_priority: bool = False,
) -> None:
    """Re-encode an OpenCV mp4v file in place as H.264 with faststart.

    OpenCV's own fourcc-based encoder selection is unreliable across
    platforms (e.g. picks unavailable hardware encoders on EC2), so outputs
    are always written as mp4v and converted here. Raises RuntimeError on
    failure.
    """
    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        logger.error(
            "ffmpeg not found on PATH or in known install locations. "
            "Output remains mp4v and will NOT play in browsers."
        )
        raise RuntimeError(
            "ffmpeg is required to produce a browser-playable video but was not found on this machine."
        )

    temp_transcode_path = path + ".h264.mp4"
    command = [
        ffmpeg_bin, "-y",
        "-i", path,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if threads:
        command += ["-threads", str(int(threads))]
    command.append(temp_transcode_path)
    # Background exports yield the CPU to live analysis jobs on Linux servers.
    if low_priority and os.name == "posix" and shutil.which("nice"):
        command = ["nice", "-n", "10", *command]
    try:
        proc = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        if not os.path.exists(temp_transcode_path) or os.path.getsize(temp_transcode_path) == 0:
            raise RuntimeError(f"ffmpeg produced no usable output. stderr: {proc.stderr}")
        os.replace(temp_transcode_path, path)
        logger.info(f"Transcoded video to H.264 using: {ffmpeg_bin}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg transcode failed (exit {e.returncode}): {e.stderr}")
        raise RuntimeError(f"ffmpeg transcode failed: {e.stderr}") from e
    finally:
        if os.path.exists(temp_transcode_path):
            try:
                os.remove(temp_transcode_path)
            except OSError:
                pass
