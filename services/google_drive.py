"""
Service for downloading and managing Google Drive files for skiing video analysis.
Supports both authenticated Google Picker access (via OAuth token) and direct shared links.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, Tuple
import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB chunk streaming


def extract_drive_file_id(url_or_id: str) -> Optional[str]:
    """
    Extract Google Drive file ID from full URL, share link, or raw ID.
    Examples:
      - https://drive.google.com/file/d/1A2B3C4D5E/view?usp=sharing
      - https://drive.google.com/open?id=1A2B3C4D5E
      - https://drive.google.com/uc?id=1A2B3C4D5E&export=download
      - 1A2B3C4D5E
    """
    if not url_or_id:
        return None

    cleaned = url_or_id.strip()

    # If it's a URL
    if "drive.google.com" in cleaned or "docs.google.com" in cleaned:
        # Pattern 1: /file/d/<file_id>
        match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", cleaned)
        if match:
            return match.group(1)

        # Pattern 2: id=<file_id>
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", cleaned)
        if match:
            return match.group(1)

        # Pattern 3: /folders/<id> or /d/<id>
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", cleaned)
        if match:
            return match.group(1)

    # If it's already a raw file ID (typically 25-50 characters: alphanumeric, dashes, underscores)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,60}", cleaned):
        return cleaned

    return None


def get_drive_file_metadata(file_id: str, oauth_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch file name and metadata from Google Drive API if oauth_token is available.
    """
    if not oauth_token:
        return {"id": file_id, "name": f"gdrive_{file_id[:8]}.mp4"}

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {"fields": "id,name,size,mimeType"}
    headers = {"Authorization": f"Bearer {oauth_token}"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.warning(
            "Failed to fetch metadata for Drive file %s (status %s): %s",
            file_id,
            response.status_code,
            response.text,
        )
    except Exception as exc:
        logger.warning("Error fetching Drive metadata for %s: %s", file_id, exc)

    return {"id": file_id, "name": f"gdrive_{file_id[:8]}.mp4"}


def download_drive_file(
    file_id: str,
    destination_path: str,
    oauth_token: Optional[str] = None,
    progress_callback=None,
) -> Tuple[str, int]:
    """
    Download file from Google Drive into destination_path.
    Returns (destination_path, total_bytes_downloaded).
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)

    if oauth_token:
        return _download_with_token(file_id, destination_path, oauth_token, progress_callback)
    return _download_public_link(file_id, destination_path, progress_callback)


def _download_with_token(
    file_id: str,
    destination_path: str,
    oauth_token: str,
    progress_callback=None,
) -> Tuple[str, int]:
    """Download using Google Drive v3 media API with Bearer token."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {oauth_token}"}

    logger.info("Downloading Drive file %s using OAuth token...", file_id)
    response = requests.get(url, headers=headers, stream=True, timeout=30)

    if response.status_code != 200:
        error_msg = response.text[:200]
        raise RuntimeError(f"Google Drive API returned HTTP {response.status_code}: {error_msg}")

    total_bytes = 0
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                if progress_callback:
                    progress_callback(total_bytes)

    logger.info("Finished downloading Drive file %s (%d bytes)", file_id, total_bytes)
    return destination_path, total_bytes


def _download_public_link(
    file_id: str,
    destination_path: str,
    progress_callback=None,
) -> Tuple[str, int]:
    """
    Download a publicly accessible or shared Google Drive file without an OAuth token.
    Handles Google's large file antivirus warning automatically.
    """
    session = requests.Session()
    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    logger.info("Downloading public Drive file %s...", file_id)
    response = session.get(url, params=params, stream=True, timeout=30)

    # Check for confirmation token on large files (> 100MB)
    token = _get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Unable to download Google Drive video (HTTP {response.status_code}). "
            "Please ensure the file link sharing is set to 'Anyone with the link can view' "
            "or use the 'Browse Google Drive' button to authenticate."
        )

    # Check content type: if Google returned HTML, it's an error page or auth wall
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        # Check if it contains virus warning form or access denied
        text_sample = response.text[:1000]
        if "Google Drive - Access denied" in text_sample or "You need access" in text_sample:
            raise RuntimeError(
                "Access denied to this Google Drive file. Please set the file's share permissions to "
                "'Anyone with the link can view' or use 'Browse Google Drive' to authorize."
            )
        # Check if form confirmation exists inside HTML
        match = re.search(r'name="confirm"\s+value="([^"]+)"', text_sample)
        if match:
            params["confirm"] = match.group(1)
            response = session.get(url, params=params, stream=True, timeout=30)

    total_bytes = 0
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                if progress_callback:
                    progress_callback(total_bytes)

    if total_bytes < 1000:
        # Extremely small file might indicate an HTML error page saved as video
        with open(destination_path, "rb") as f:
            start_bytes = f.read(200).decode("utf-8", errors="ignore")
        if "<!DOCTYPE html" in start_bytes or "<html" in start_bytes:
            os.remove(destination_path)
            raise RuntimeError(
                "Google Drive returned an HTML page instead of a video. "
                "Please verify the file is shared with 'Anyone with the link' or authenticate via 'Browse Google Drive'."
            )

    logger.info("Finished downloading public Drive file %s (%d bytes)", file_id, total_bytes)
    return destination_path, total_bytes


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    """Retrieve Google Drive download confirmation cookie token for large files."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

