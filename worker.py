"""Background worker: runs video analysis jobs one at a time, outside the API.

Start it next to the API when JOB_RUNNER=worker:

    python worker.py

It only claims jobs that were created for the worker (runner="worker"), so it
never touches jobs that belong to an API running with JOB_RUNNER=inline. On
SIGTERM/Ctrl+C it stops taking new jobs and finishes the current one.
"""

import logging
import os
import signal
import threading
import time

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bluerun.worker")

from database import SessionLocal, close_session_quietly  # noqa: E402
from services import job_queue  # noqa: E402
from services.bootstrap import check_schema_on_startup  # noqa: E402
from services.video_exports import claim_pending_requests, recover_stalled_exports  # noqa: E402

POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "3"))
_stop = threading.Event()


def _request_stop(signum, _frame):
    logger.info("Signal %s received: finishing the current job, then stopping", signum)
    _stop.set()


def _next_job():
    db = SessionLocal()
    try:
        return job_queue.claim_next_job(db)
    finally:
        close_session_quietly(db)


def run_forever() -> None:
    check_schema_on_startup()
    db = SessionLocal()
    try:
        recovered = job_queue.recover_interrupted_jobs(db)
        if recovered:
            logger.info("Recovered %s interrupted job(s)", recovered)
    finally:
        close_session_quietly(db)

    logger.info("Worker started (polling every %ss)", POLL_SECONDS)
    last_watchdog = 0.0
    while not _stop.is_set():
        if time.time() - last_watchdog > 60:
            last_watchdog = time.time()
            try:
                recover_stalled_exports()
            except Exception as exc:
                logger.warning("Stalled-export check failed: %s", exc)
        try:
            job = _next_job()
            if job:
                job_id, kind, payload = job
                logger.info("Running job %s (%s)", job_id, kind)
                job_queue.run_job(job_id, kind, payload)
                continue
            # No analysis waiting: start any download versions people asked for.
            claim_pending_requests()
        except Exception as exc:
            logger.exception("Worker loop error: %s", exc)
        _stop.wait(POLL_SECONDS)
    logger.info("Worker stopped")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    run_forever()
