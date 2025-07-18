# services/file_watcher.py

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import httpx  # To call FastAPI endpoint
from pathlib import Path

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']

API_URL = "http://127.0.0.1:8000/analyze/"  # Your FastAPI endpoint
DEFAULT_PERSON_ID = 1  # Optional default

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and self._is_video_file(event.src_path):
            print(f"New video detected: {event.src_path}")
            try:
                time.sleep(5)  # Wait to ensure file is fully written
                self.call_fastapi_analysis(event.src_path)
            except Exception as e:
                print(f"Error analyzing video: {e}")

    def _is_video_file(self, filename):
        _, ext = os.path.splitext(filename)
        return ext.lower() in VIDEO_EXTENSIONS

    def call_fastapi_analysis(self, file_path):
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as video_file:
            files = {"file": (filename, video_file, "video/mp4")}
            data = {"person_id": DEFAULT_PERSON_ID}  # Optional, set default
            print("Sending video to FastAPI /analyze endpoint...")
            response = httpx.post(API_URL, data=data, files=files)

        if response.status_code == 200:
            print("✅ Analysis result:", response.json())
        else:
            print("❌ Failed to analyze video. Status code:", response.status_code)
            print("Response:", response.text)

def start_watching():
    # Adjust path to go up to root and then watch /uploads outside services/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_to_watch = os.path.join(base_dir, "uploads")

    if not os.path.exists(folder_to_watch):
        os.makedirs(folder_to_watch)

    print(f"Started watching directory: {folder_to_watch}")
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopped watching.")
    observer.join()

if __name__ == "__main__":
    start_watching()
