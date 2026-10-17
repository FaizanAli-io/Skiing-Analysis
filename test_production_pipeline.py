"""Worker queue, on-demand download versions, direct uploads and migrations.

Uses the in-memory SQLite database from test_account_and_sharing; nothing
here touches the real database, S3 or video processing.
"""

import os
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import database
from test_account_and_sharing import TestingSession, _Fixture, engine, override_get_db, wait_for_export_threads
from models.analysis_job import AnalysisJob
from models.video_analysis import VideoAnalysis
from services import job_queue, video_exports


def _wait_for(predicate, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


class WorkerQueueTests(_Fixture):
    def _job(self, job_id, runner, status="pending", attempts=0, created=None, host=None):
        db = TestingSession()
        db.add(AnalysisJob(
            id=job_id, person_id=self.ids["alice"], file_name="v.mp4", status=status,
            runner=runner, job_kind="upload", payload={"file_path": "/tmp/v.mp4", "user_id": self.ids["alice"]},
            attempts=attempts, created_at=created or datetime.utcnow(), runner_host=host or job_queue.this_host(),
        ))
        db.commit()
        db.close()

    def test_worker_claims_only_worker_jobs_once(self):
        self._job("inline-1", job_queue.RUNNER_INLINE, created=datetime(2026, 1, 1))
        self._job("worker-1", job_queue.RUNNER_WORKER, created=datetime(2026, 1, 2))
        db = TestingSession()
        claimed = job_queue.claim_next_job(db)
        self.assertEqual(claimed[0], "worker-1")
        self.assertEqual(claimed[1], "upload")
        self.assertIsNone(job_queue.claim_next_job(db))
        job = db.query(AnalysisJob).filter(AnalysisJob.id == "worker-1").first()
        self.assertEqual((job.status, job.attempts), ("processing", 1))
        inline = db.query(AnalysisJob).filter(AnalysisJob.id == "inline-1").first()
        self.assertEqual(inline.status, "pending")
        db.close()

    def test_worker_ignores_jobs_created_on_another_machine(self):
        self._job("prod-job", job_queue.RUNNER_WORKER, host="ip-10-0-0-5")
        db = TestingSession()
        self.assertIsNone(job_queue.claim_next_job(db))
        self.assertEqual(job_queue.recover_interrupted_jobs(db), 0)
        db.close()

    def test_interrupted_jobs_retry_once_then_fail(self):
        self._job("retry-me", job_queue.RUNNER_WORKER, status="processing", attempts=1)
        self._job("give-up", job_queue.RUNNER_WORKER, status="processing", attempts=2)
        self._job("inline-busy", job_queue.RUNNER_INLINE, status="processing", attempts=1)
        db = TestingSession()
        job_queue.recover_interrupted_jobs(db)
        states = {job.id: job.status for job in db.query(AnalysisJob).all()}
        db.close()
        self.assertEqual(states["retry-me"], "pending")
        self.assertEqual(states["give-up"], "failed")
        self.assertEqual(states["inline-busy"], "processing")

    def test_dispatch_runs_inline_or_leaves_job_for_worker(self):
        class FakeTasks:
            def __init__(self):
                self.calls = []

            def add_task(self, *args, **kwargs):
                self.calls.append(args)

        for mode, expected_calls in (("inline", 1), ("worker", 0)):
            tasks = FakeTasks()
            with mock.patch.dict(os.environ, {"JOB_RUNNER": mode}), \
                    mock.patch("database.SessionLocal", TestingSession):
                job_queue.queue_job(tasks, f"job-{mode}", "upload", self.ids["alice"], "v.mp4", "athlete", True, {"file_path": "x"})
            self.assertEqual(len(tasks.calls), expected_calls)
            db = TestingSession()
            job = db.query(AnalysisJob).filter(AnalysisJob.id == f"job-{mode}").first()
            self.assertEqual((job.runner, job.job_kind, job.payload), (mode, "upload", {"file_path": "x"}))
            db.close()

    def test_secrets_are_scrubbed_from_job_payload(self):
        db = TestingSession()
        db.add(AnalysisJob(id="drive-1", person_id=self.ids["alice"], file_name="d.mp4", status="processing",
                           runner="worker", job_kind="drive", payload={"oauth_token": "secret", "drive_file_id": "abc"}))
        db.commit()
        db.close()
        with mock.patch("database.SessionLocal", TestingSession):
            job_queue.scrub_payload("drive-1", "oauth_token")
        db = TestingSession()
        self.assertEqual(db.query(AnalysisJob).filter(AnalysisJob.id == "drive-1").first().payload, {"drive_file_id": "abc"})
        db.close()

    def test_upload_names_cannot_escape_the_temp_folder(self):
        self.assertEqual(job_queue.safe_upload_name("../../etc/passwd", "x.mp4"), "passwd.mp4")
        self.assertEqual(job_queue.safe_upload_name("My Run 1.MOV", "x.mp4"), "My_Run_1.MOV")


class OnDemandExportTests(_Fixture):
    def setUp(self):
        super().setUp()
        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update({"video_exports": None})
        db.commit()
        db.close()
        self.source = os.path.join(self.tmpdir.name, "source.mp4")
        self.layer = os.path.join(self.tmpdir.name, "layer.avi")
        for path in (self.source, self.layer):
            with open(path, "wb") as handle:
                handle.write(b"x")
        self.manifest = {
            "display_mode": "athlete",
            "source_path": self.source,
            "overlay_path": self.layer,
            "base": os.path.join(self.tmpdir.name, "run"),
            "processed_indices": [0],
        }
        self.rendered = []

        def fake_render(manifest, mode, fmt, progress=None):
            self.rendered.append((mode, fmt))
            if progress:
                progress(0.5)
            path = os.path.join(self.tmpdir.name, f"{mode}_{fmt}_{len(self.rendered)}.mp4")
            with open(path, "wb") as handle:
                handle.write(b"video")
            return path

        self.patches = [
            mock.patch("database.SessionLocal", TestingSession),
            mock.patch("services.notifications.SessionLocal", TestingSession),
            mock.patch.object(video_exports, "render_variant", side_effect=fake_render),
            mock.patch("services.aws_s3.S3Manager.is_enabled", return_value=False),
            mock.patch.dict(os.environ, {"JOB_RUNNER": "inline", "EXPORTS_EAGER": "athlete:vertical"}),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        wait_for_export_threads()
        for patcher in reversed(self.patches):
            patcher.stop()
        super().tearDown()

    def _run(self):
        db = TestingSession()
        run = db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).first()
        db.expunge(run)
        db.close()
        return run

    def test_only_eager_version_is_made_and_others_are_preparable(self):
        video_exports.start_exports(self.ids["run"], self.manifest)
        wait_for_export_threads()
        run = self._run()
        self.assertEqual(run.exports_status, "ready")
        self.assertEqual(self.rendered, [("athlete", "vertical")])
        self.assertEqual(run.available_downloads["athlete"], ["original", "vertical"])
        self.assertEqual(run.preparable_downloads, {"athlete": ["square"], "coach": ["original", "vertical", "square"]})
        self.assertEqual(run.export_source["storage"], "local")
        self.assertTrue(os.path.exists(self.source))  # kept for later versions

    def test_prepare_endpoint_renders_requested_version(self):
        from routes import video_analysis

        video_exports.start_exports(self.ids["run"], self.manifest)
        wait_for_export_threads()
        self.assertEqual(self._run().exports_status, "ready")

        app = FastAPI()
        app.include_router(video_analysis.router, prefix="/api")
        app.dependency_overrides[database.get_db] = override_get_db
        client = TestClient(app)
        with mock.patch("services.auth.SessionLocal", TestingSession):
            from routes import auth as auth_routes

            auth_app = FastAPI()
            auth_app.include_router(auth_routes.router, prefix="/api")
            auth_app.dependency_overrides[database.get_db] = override_get_db
            login = TestClient(auth_app).post("/api/auth/login", json={"email": "alice@test.com", "password": "alice-pass-1"})
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

            first = client.post(f"/api/videos/{self.ids['run']}/prepare?mode=coach&format=square", headers=headers)
            self.assertEqual(first.json(), {"status": "preparing"})
            # Let the background render finish before reading the shared test DB.
            wait_for_export_threads()
            self.assertIn("square", self._run().available_downloads["coach"])
            again = client.post(f"/api/videos/{self.ids['run']}/prepare?mode=coach&format=square", headers=headers)
            self.assertEqual(again.json(), {"status": "ready"})

            # Whoever asked is told it's ready, for that exact version.
            from models.notification import Notification

            db = TestingSession()
            notes = db.query(Notification).filter(Notification.person_id == self.ids["alice"]).all()
            self.assertEqual([(n.kind, n.title, n.video_mode, n.video_format) for n in notes],
                             [("video_ready", "Your Coach · Square video is ready", "coach", "square")])
            run = db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).first()
            self.assertFalse(run.export_requesters)
            self.assertFalse(run.export_progress)
            db.close()
            status = client.get(f"/api/videos/{self.ids['run']}/download-status", headers=headers).json()
            self.assertIn("square", status["available_downloads"]["coach"])

            bob = TestClient(auth_app).post("/api/auth/login", json={"email": "bob@test.com", "password": "bob-pass-1"})
            other = client.post(
                f"/api/videos/{self.ids['run']}/prepare?mode=coach&format=vertical",
                headers={"Authorization": f"Bearer {bob.json()['access_token']}"},
            )
            self.assertEqual(other.status_code, 404)
        self.assertEqual(self.rendered.count(("coach", "square")), 1)

    def _stall(self, minutes_ago, attempts=None):
        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update({
            "exports_status": "processing",
            "exports_updated_at": datetime.utcnow() - timedelta(minutes=minutes_ago),
            "export_requests": ["coach:square"],
            "export_requesters": {"coach:square": [self.ids["alice"]]},
            "export_progress": {"coach:square": 0},
            "export_attempts": attempts,
        })
        db.commit()
        db.close()

    def test_stalled_render_is_requeued_but_live_one_is_left_alone(self):
        with mock.patch.dict(os.environ, {"JOB_RUNNER": "worker"}):  # no inline thread
            self._stall(minutes_ago=1)
            self.assertEqual(video_exports.recover_stalled_exports(), 0)
            self.assertEqual(self._run().exports_status, "processing")

            self._stall(minutes_ago=10)
            self.assertEqual(video_exports.recover_stalled_exports(), 1)
        run = self._run()
        self.assertEqual((run.exports_status, run.export_attempts), ("requested", 1))
        self.assertEqual(run.export_requests, ["coach:square"])
        self.assertFalse(run.export_progress)

    def test_repeatedly_stalled_render_fails_and_tells_the_requester(self):
        from models.notification import Notification

        with mock.patch.dict(os.environ, {"JOB_RUNNER": "worker"}):
            self._stall(minutes_ago=10, attempts=video_exports.MAX_RENDER_ATTEMPTS - 1)
            self.assertEqual(video_exports.recover_stalled_exports(), 1)
        run = self._run()
        self.assertEqual(run.exports_status, "failed")
        self.assertFalse(run.export_requests)
        db = TestingSession()
        notes = db.query(Notification).filter(Notification.person_id == self.ids["alice"]).all()
        self.assertEqual([(n.kind, n.video_format) for n in notes], [("video_failed", "square")])
        db.close()

    def test_requeued_render_runs_again_inline(self):
        self._stall(minutes_ago=10)
        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update({
            "export_source": {
                "storage": "local", "source_path": self.source, "layer_path": self.layer,
                "manifest": {"display_mode": "athlete", "processed_indices": [0]},
                "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            },
        })
        db.commit()
        db.close()
        video_exports.recover_stalled_exports()
        wait_for_export_threads()
        run = self._run()
        self.assertIn("square", run.available_downloads["coach"])
        self.assertEqual(run.exports_status, "ready")
        self.assertIsNone(run.export_attempts)

    def test_expired_source_cannot_be_prepared(self):
        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update({
            "export_source": {"storage": "local", "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat()},
        })
        db.commit()
        db.close()
        self.assertEqual(self._run().preparable_downloads, {"athlete": [], "coach": []})


class DirectUploadTests(_Fixture):
    def setUp(self):
        super().setUp()
        from routes import auth as auth_routes, uploads

        self.app = FastAPI()
        self.app.include_router(uploads.router, prefix="/api")
        self.app.include_router(auth_routes.router, prefix="/api")
        self.app.dependency_overrides[database.get_db] = override_get_db
        self.client = TestClient(self.app)
        self.patches = [
            mock.patch("routes.uploads.SessionLocal", TestingSession),
            mock.patch("database.SessionLocal", TestingSession),
            mock.patch("services.aws_s3.S3Manager.is_enabled", return_value=True),
            mock.patch.dict(os.environ, {"DIRECT_S3_UPLOADS": "true", "MAX_UPLOAD_MB": "100", "JOB_RUNNER": "worker"}),
        ]
        for patcher in self.patches:
            patcher.start()
        login = self.client.post("/api/auth/login", json={"email": "admin@test.com", "password": "admin-pass-1"})
        self.admin = {"Authorization": f"Bearer {login.json()['access_token']}"}

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        super().tearDown()

    def test_presign_validates_and_scopes_the_key(self):
        signed = {"url": "https://bucket.s3.amazonaws.com/", "fields": {"key": "k"}}
        with mock.patch("services.aws_s3.S3Manager.presign_upload", return_value=signed) as presign:
            ok = self.client.post("/api/uploads/presign", headers=self.admin,
                                  json={"file_name": "../My Run.MOV", "content_type": "video/quicktime", "size": 5_000_000})
            self.assertEqual(ok.status_code, 200, ok.text)
            key = ok.json()["key"]
            self.assertTrue(key.startswith("uploads/") and key.endswith("/My_Run.MOV"))
            self.assertEqual(presign.call_args.args[1], "video/quicktime")
            too_big = self.client.post("/api/uploads/presign", headers=self.admin,
                                       json={"file_name": "a.mp4", "content_type": "video/mp4", "size": 500 * 1024 * 1024})
            self.assertEqual(too_big.status_code, 422)
            not_video = self.client.post("/api/uploads/presign", headers=self.admin,
                                         json={"file_name": "a.pdf", "content_type": "application/pdf", "size": 10})
            self.assertEqual(not_video.status_code, 422)
        self.assertEqual(self.client.post("/api/uploads/presign", json={}).status_code, 401)

    def test_analyze_s3_checks_object_and_queues_worker_job(self):
        bad = self.client.post("/api/analyze-s3/", headers=self.admin, json={"s3_key": "videos/other.mp4", "user_id": self.ids["alice"]})
        self.assertEqual(bad.status_code, 422)
        with mock.patch("services.aws_s3.S3Manager.object_size", return_value=None):
            missing = self.client.post("/api/analyze-s3/", headers=self.admin,
                                       json={"s3_key": "uploads/abc/run.mp4", "user_id": self.ids["alice"]})
            self.assertEqual(missing.status_code, 404)
        with mock.patch("services.aws_s3.S3Manager.object_size", return_value=1234):
            queued = self.client.post("/api/analyze-s3/", headers=self.admin,
                                      json={"s3_key": "uploads/abc/run.mp4", "user_id": self.ids["alice"], "report": True})
        self.assertEqual(queued.status_code, 200, queued.text)
        db = TestingSession()
        job = db.query(AnalysisJob).filter(AnalysisJob.id == queued.json()["job_id"]).first()
        self.assertEqual((job.runner, job.job_kind, job.status), ("worker", "s3", "pending"))
        self.assertEqual(job.payload["s3_key"], "uploads/abc/run.mp4")
        self.assertTrue(os.path.isabs(job.payload["file_path"]))
        db.close()

    def test_config_reports_direct_upload_flag(self):
        self.assertTrue(self.client.get("/api/config/uploads").json()["direct_s3"])
        with mock.patch.dict(os.environ, {"DIRECT_S3_UPLOADS": "false"}):
            self.assertFalse(self.client.get("/api/config/uploads").json()["direct_s3"])


class MigrationTests(unittest.TestCase):
    def test_migrations_record_version_and_startup_check_is_quiet(self):
        from services import bootstrap

        with mock.patch.object(bootstrap, "engine", engine), \
                mock.patch.object(bootstrap, "SessionLocal", TestingSession), \
                mock.patch.dict(os.environ, {"DEFAULT_ADMIN_EMAIL": "", "DEFAULT_ADMIN_PASSWORD": ""}):
            database.Base.metadata.drop_all(bind=engine)
            version = bootstrap.run_migrations()
            self.assertEqual(bootstrap.stored_fingerprint(), version)
            with mock.patch.object(bootstrap, "run_migrations") as rerun:
                bootstrap.check_schema_on_startup()
                rerun.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class NotificationTests(_Fixture):
    def setUp(self):
        super().setUp()
        from routes import auth as auth_routes, notifications

        app = FastAPI()
        app.include_router(notifications.router, prefix="/api")
        app.include_router(auth_routes.router, prefix="/api")
        app.dependency_overrides[database.get_db] = override_get_db
        self.client = TestClient(app)
        self.session_patch = mock.patch("services.notifications.SessionLocal", TestingSession)
        self.session_patch.start()

    def tearDown(self):
        self.session_patch.stop()
        super().tearDown()

    def headers(self, email, password):
        token = self.client.post("/api/auth/login", json={"email": email, "password": password}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_run_ready_notifies_athlete_and_admins(self):
        from services.notifications import notify_run_ready

        notify_run_ready(self.ids["run"])
        alice = self.client.get("/api/notifications", headers=self.headers("alice@test.com", "alice-pass-1")).json()
        self.assertEqual(alice["unread"], 1)
        self.assertEqual(alice["items"][0]["title"], "Your new run is ready!")
        self.assertTrue(alice["items"][0]["body"].startswith("Run 1 · Blue IQ"))
        admin = self.client.get("/api/notifications", headers=self.headers("admin@test.com", "admin-pass-1")).json()
        self.assertEqual(admin["items"][0]["title"], "Alice's run 1 is ready")
        bob = self.client.get("/api/notifications", headers=self.headers("bob@test.com", "bob-pass-1")).json()
        self.assertEqual(bob["items"], [])

    def test_read_download_and_dismiss_flow(self):
        from services.notifications import notify_run_ready

        notify_run_ready(self.ids["run"])
        alice = self.headers("alice@test.com", "alice-pass-1")
        item_id = self.client.get("/api/notifications", headers=alice).json()["items"][0]["id"]

        bob = self.headers("bob@test.com", "bob-pass-1")
        self.assertEqual(self.client.post(f"/api/notifications/{item_id}/dismiss", headers=bob).status_code, 404)

        self.client.post("/api/notifications/read-all", headers=alice)
        self.assertEqual(self.client.get("/api/notifications", headers=alice).json()["unread"], 0)
        self.client.post(f"/api/notifications/{item_id}/downloaded", headers=alice)
        self.assertTrue(self.client.get("/api/notifications", headers=alice).json()["items"][0]["downloaded"])
        self.client.post(f"/api/notifications/{item_id}/dismiss", headers=alice)
        self.assertEqual(self.client.get("/api/notifications", headers=alice).json()["items"], [])

    def test_processing_jobs_show_progress_to_the_right_people(self):
        db = TestingSession()
        for job_id, person, progress in (("a-job", self.ids["alice"], 45), ("b-job", self.ids["bob"], 10)):
            db.add(AnalysisJob(id=job_id, person_id=person, file_name=f"{job_id}.mp4", status="processing", progress=progress))
        db.add(AnalysisJob(id="done-job", person_id=self.ids["alice"], file_name="old.mp4", status="completed", progress=100))
        db.commit()
        db.close()
        alice = self.client.get("/api/notifications", headers=self.headers("alice@test.com", "alice-pass-1")).json()
        self.assertEqual([(j["job_id"], j["progress"]) for j in alice["active_jobs"]], [("a-job", 45)])
        self.assertIsNone(alice["active_jobs"][0]["file_name"])
        admin = self.client.get("/api/notifications", headers=self.headers("admin@test.com", "admin-pass-1")).json()
        self.assertEqual(sorted(j["athlete_name"] for j in admin["active_jobs"]), ["Alice", "Bob"])

    def test_failed_upload_notifies_admins_only(self):
        from services.notifications import notify_run_failed

        db = TestingSession()
        db.add(AnalysisJob(id="bad-job", person_id=self.ids["alice"], file_name="broken.mp4", status="failed"))
        db.commit()
        db.close()
        notify_run_failed("bad-job")
        admin = self.client.get("/api/notifications", headers=self.headers("admin@test.com", "admin-pass-1")).json()
        self.assertEqual(admin["items"][0]["title"], "Upload for Alice failed")
        alice = self.client.get("/api/notifications", headers=self.headers("alice@test.com", "alice-pass-1")).json()
        self.assertEqual(alice["items"], [])

    def test_bell_lists_versions_being_prepared_for_the_requester(self):
        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update({
            "exports_status": "processing",
            "exports_updated_at": datetime.utcnow(),
            "export_requests": ["coach:square"],
            "export_requesters": {"coach:square": [self.ids["alice"]]},
            "export_progress": {"coach:square": 40},
        })
        db.commit()
        db.close()
        alice = self.client.get("/api/notifications", headers=self.headers("alice@test.com", "alice-pass-1")).json()
        self.assertEqual(alice["preparing_videos"], [{
            "analysis_id": self.ids["run"], "mode": "coach", "format": "square",
            "label": "Coach · Square", "progress": 40,
        }])
        admin = self.client.get("/api/notifications", headers=self.headers("admin@test.com", "admin-pass-1")).json()
        self.assertEqual(admin["preparing_videos"], [])



class StagedPreviewTests(_Fixture):
    """Videos browsers can't play (HEVC .MOV) are previewed from a server frame."""

    def setUp(self):
        super().setUp()
        import subprocess
        from routes import auth as auth_routes, uploads
        from services.video_encoding import find_ffmpeg

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            self.skipTest("ffmpeg not installed")
        self.mov = os.path.join(self.tmpdir.name, "iphone.MOV")
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=2",
             "-c:v", "libx265", "-tag:v", "hvc1", self.mov],
            check=True, capture_output=True,
        )
        self.temp_dir = os.path.join(self.tmpdir.name, "temp_videos")
        os.makedirs(self.temp_dir)

        app = FastAPI()
        app.include_router(uploads.router, prefix="/api")
        app.include_router(auth_routes.router, prefix="/api")
        app.dependency_overrides[database.get_db] = override_get_db
        self.client = TestClient(app)
        self.patches = [
            mock.patch("routes.uploads.SessionLocal", TestingSession),
            mock.patch("routes.uploads.TEMP_VIDEO_DIR", self.temp_dir),
            mock.patch("database.SessionLocal", TestingSession),
            mock.patch.dict(os.environ, {"JOB_RUNNER": "worker"}),
        ]
        for patcher in self.patches:
            patcher.start()
        login = self.client.post("/api/auth/login", json={"email": "admin@test.com", "password": "admin-pass-1"})
        self.admin = {"Authorization": f"Bearer {login.json()['access_token']}"}

    def tearDown(self):
        for patcher in reversed(getattr(self, "patches", [])):
            patcher.stop()
        super().tearDown()

    def test_stage_returns_preview_and_queue_reuses_the_file(self):
        with open(self.mov, "rb") as handle:
            staged = self.client.post("/api/uploads/stage", headers=self.admin,
                                      files={"file": ("Daniel B.MOV", handle, "video/quicktime")})
        self.assertEqual(staged.status_code, 200, staged.text)
        body = staged.json()
        self.assertTrue(body["preview"].startswith("data:image/jpeg;base64,"))
        self.assertEqual((body["width"], body["height"]), (320, 180))
        self.assertEqual(len(os.listdir(self.temp_dir)), 1)  # only the staged video, preview cleaned up

        queued = self.client.post("/api/analyze-staged/", headers=self.admin, json={
            "staging_id": body["staging_id"], "user_id": self.ids["alice"], "rotation": "-90", "file_name": "Daniel B.MOV",
        })
        self.assertEqual(queued.status_code, 200, queued.text)
        db = TestingSession()
        job = db.query(AnalysisJob).filter(AnalysisJob.id == queued.json()["job_id"]).first()
        self.assertEqual((job.job_kind, job.runner, job.file_name), ("upload", "worker", "Daniel B.MOV"))
        self.assertEqual(job.payload["rotation_override"], -90)
        self.assertTrue(os.path.exists(job.payload["file_path"]))
        db.close()
        again = self.client.post("/api/analyze-staged/", headers=self.admin,
                                 json={"staging_id": body["staging_id"], "user_id": self.ids["alice"]})
        self.assertEqual(again.status_code, 404)  # staged file was moved to the job

    def test_rejects_non_videos_and_bad_ids(self):
        bad = self.client.post("/api/uploads/stage", headers=self.admin,
                               files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")})
        self.assertEqual(bad.status_code, 422)
        broken = self.client.post("/api/uploads/stage", headers=self.admin,
                                  files={"file": ("broken.mov", b"not really a video", "video/quicktime")})
        self.assertEqual(broken.status_code, 422)
        self.assertEqual(os.listdir(self.temp_dir), [])
        self.assertEqual(self.client.post("/api/analyze-staged/", headers=self.admin,
                                          json={"staging_id": "../../etc", "user_id": self.ids["alice"]}).status_code, 404)
        self.assertEqual(self.client.post("/api/uploads/stage", files={"file": ("a.mov", b"x", "video/quicktime")}).status_code, 401)
