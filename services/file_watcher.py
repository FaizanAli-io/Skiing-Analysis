# services/file_watcher.py

import os
import time
import threading
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import httpx  # To call FastAPI endpoint
from pathlib import Path

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']

API_URL = "http://127.0.0.1:8000/analyze/"  # Your FastAPI endpoint
DEFAULT_PERSON_ID = 1  # Optional default

# Configure logging
def setup_logging():
    """Configure logging for the file watcher service"""
    # Create logs directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, "file_watcher.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class VideoHandler(FileSystemEventHandler):
    def __init__(self):
        self.pending_files = {}  # Track files being written
        logger.info("VideoHandler initialized - ready to monitor video files")
        
    def on_created(self, event):
        if not event.is_directory and self._is_video_file(event.src_path):
            logger.info(f"New video detected: {event.src_path}")
            # Start monitoring this file in a separate thread
            threading.Thread(
                target=self._monitor_file_completion, 
                args=(event.src_path,),
                daemon=True
            ).start()
    
    def on_modified(self, event):
        # Handle cases where file is created then modified (some systems)
        if not event.is_directory and self._is_video_file(event.src_path):
            if event.src_path not in self.pending_files:
                logger.info(f"Video modification detected: {event.src_path}")
                threading.Thread(
                    target=self._monitor_file_completion, 
                    args=(event.src_path,),
                    daemon=True
                ).start()

    def _is_video_file(self, filename):
        _, ext = os.path.splitext(filename)
        is_video = ext.lower() in VIDEO_EXTENSIONS
        if is_video:
            logger.debug(f"File {filename} identified as video file with extension {ext}")
        return is_video

    def _monitor_file_completion(self, file_path):
        """Monitor file until it's completely written"""
        if file_path in self.pending_files:
            logger.debug(f"File {file_path} already being monitored, skipping duplicate monitoring")
            return  # Already monitoring this file
            
        self.pending_files[file_path] = True
        filename = os.path.basename(file_path)
        logger.info(f"Starting file completion monitoring for: {filename}")
        
        try:
            previous_size = 0
            stable_count = 0
            max_wait_time = 300  # Maximum 5 minutes wait
            start_time = time.time()
            
            logger.info(f"Monitoring file completion: {filename} (max wait: {max_wait_time}s)")
            
            while time.time() - start_time < max_wait_time:
                try:
                    # Check if file exists and is accessible
                    if not os.path.exists(file_path):
                        logger.debug(f"File {filename} does not exist yet, waiting...")
                        time.sleep(1)
                        continue
                    
                    # Try to get file size
                    current_size = os.path.getsize(file_path)
                    logger.debug(f"File {filename} current size: {current_size} bytes (previous: {previous_size})")
                    
                    # Check if file is still being written by trying to open it
                    if self._is_file_ready(file_path):
                        if current_size == previous_size and current_size > 0:
                            stable_count += 1
                            logger.debug(f"File {filename} stable for {stable_count} checks")
                            # File size hasn't changed for 3 consecutive checks
                            if stable_count >= 3:
                                logger.info(f"File write completed: {filename} ({current_size} bytes)")
                                break
                        else:
                            stable_count = 0
                            previous_size = current_size
                            logger.debug(f"File {filename} size changed, resetting stable count")
                    else:
                        stable_count = 0
                        logger.debug(f"File {filename} not ready (still locked)")
                    
                    time.sleep(2)  # Check every 2 seconds
                    
                except (OSError, IOError) as e:
                    # File might still be locked
                    logger.warning(f"File access error for {filename} (still writing?): {e}")
                    time.sleep(2)
                    continue
            
            elapsed_time = time.time() - start_time
            
            # File is ready or timeout reached
            if elapsed_time >= max_wait_time:
                logger.warning(f"Timeout reached for {filename} after {elapsed_time:.1f}s, proceeding anyway")
            else:
                logger.info(f"File {filename} ready after {elapsed_time:.1f}s")
            
            # Final check before analysis
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                final_size = os.path.getsize(file_path)
                logger.info(f"Starting analysis for {filename} (final size: {final_size} bytes)")
                self.call_fastapi_analysis(file_path)
            else:
                logger.error(f"File {filename} is empty or doesn't exist, skipping analysis")
                
        except Exception as e:
            logger.error(f"Error monitoring file completion for {filename}: {e}", exc_info=True)
        finally:
            # Remove from pending files
            self.pending_files.pop(file_path, None)
            logger.debug(f"Stopped monitoring file: {filename}")

    def _is_file_ready(self, file_path):
        """Check if file is ready by trying to open it exclusively"""
        filename = os.path.basename(file_path)
        try:
            # Try to open file in append mode - this will fail if file is locked
            with open(file_path, 'rb') as f:
                # Try to read a small chunk to ensure file is accessible
                f.read(1024)
            logger.debug(f"File {filename} is ready and accessible")
            return True
        except (OSError, IOError, PermissionError) as e:
            logger.debug(f"File {filename} not ready: {e}")
            return False

    def call_fastapi_analysis(self, file_path):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        logger.info(f"Starting FastAPI analysis for: {filename} ({file_size} bytes)")
        
        try:
            with open(file_path, "rb") as video_file:
                files = {"file": (filename, video_file, "video/mp4")}
                data = {"person_id": DEFAULT_PERSON_ID}
                
                logger.info(f"Sending POST request to {API_URL} for file: {filename}")
                start_time = time.time()
                
                response = httpx.post(API_URL, data=data, files=files, timeout=300)
                
                response_time = time.time() - start_time
                logger.info(f"FastAPI response received in {response_time:.2f}s for {filename}")

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Analysis successful for {filename}: {result}")
            else:
                logger.error(f"Analysis failed for {filename}. Status: {response.status_code}")
                logger.error(f"Response body: {response.text}")
                
        except httpx.TimeoutException:
            logger.error(f"Timeout error calling FastAPI analysis for {filename}")
        except httpx.RequestError as e:
            logger.error(f"Request error calling FastAPI analysis for {filename}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling FastAPI analysis for {filename}: {e}", exc_info=True)

def start_watching():
    """Start watching the uploads directory for new video files"""
    # Adjust path to go up to root and then watch /uploads outside services/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_to_watch = os.path.join(base_dir, "uploads")

    logger.info(f"Initializing file watcher service")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Target watch folder: {folder_to_watch}")

    if not os.path.exists(folder_to_watch):
        logger.info(f"Creating uploads directory: {folder_to_watch}")
        os.makedirs(folder_to_watch)
    else:
        logger.info(f"Uploads directory already exists: {folder_to_watch}")

    # Log current files in directory
    try:
        existing_files = os.listdir(folder_to_watch)
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing files in watch directory")
            for file in existing_files:
                logger.debug(f"Existing file: {file}")
        else:
            logger.info("Watch directory is empty")
    except Exception as e:
        logger.error(f"Error listing existing files: {e}")

    logger.info(f"Starting file system observer for: {folder_to_watch}")
    logger.info(f"Supported video extensions: {VIDEO_EXTENSIONS}")
    
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    
    try:
        observer.start()
        logger.info("File watcher started successfully - monitoring for video files")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - stopping file watcher")
        observer.stop()
    except Exception as e:
        logger.error(f"Unexpected error in file watcher: {e}", exc_info=True)
        observer.stop()
        raise
    finally:
        logger.info("Waiting for observer to stop...")
        observer.join()
        logger.info("File watcher stopped successfully")

if __name__ == "__main__":
    logger.info("=== SKI ANALYZER FILE WATCHER SERVICE STARTING ===")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"API URL: {API_URL}")
    logger.info(f"Default Person ID: {DEFAULT_PERSON_ID}")
    
    try:
        start_watching()
    except Exception as e:
        logger.critical(f"Critical error starting file watcher service: {e}", exc_info=True)
        exit(1)