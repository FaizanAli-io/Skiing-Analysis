"""Account security, download access, client data visibility and coach notes.

Runs against an in-memory SQLite database (never the real one) with email
sending stubbed out.
"""

import os
import tempfile
from datetime import datetime
import unittest
from unittest import mock

os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ["AI_COACH_NOTES_ENABLED"] = "false"
# Never send real email from tests (set before .env is loaded; dotenv won't override).
os.environ["SMTP_USERNAME"] = ""
os.environ["SMTP_PASSWORD"] = ""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database
import models  # noqa: F401  (registers every table)
import services.auth as auth_service
from models.person import Person
from models.video_analysis import VideoAnalysis
from routes import auth, client_portal, video_analysis
from services.auth import hash_password
from services.coach_notes import fallback_coach_note, generate_coach_note, validate_coach_note


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def wait_for_export_threads(timeout=30):
    """Background export threads must not outlive a test (run IDs repeat)."""
    import threading
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        # Per-job finalize/request threads; the pool's own idle threads live on.
        busy = [t for t in threading.enumerate() if t.name.startswith(("video-export-finalize", "video-export-request"))]
        if not busy:
            return
        time.sleep(0.05)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()
for router in (auth.router, video_analysis.router, client_portal.router):
    app.include_router(router, prefix="/api")
app.dependency_overrides[database.get_db] = override_get_db


class _Fixture(unittest.TestCase):
    """Fresh database with an admin, two clients and one of Alice's runs."""

    def setUp(self):
        database.Base.metadata.drop_all(bind=engine)
        database.Base.metadata.create_all(bind=engine)
        self.session_patch = mock.patch.object(auth_service, "SessionLocal", TestingSession)
        self.session_patch.start()
        self.client = TestClient(app)
        self.tmpdir = tempfile.TemporaryDirectory()

        db = TestingSession()
        self.admin = Person(name="Admin", email="admin@test.com", password_hash=hash_password("admin-pass-1"), role="admin")
        self.alice = Person(name="Alice", email="alice@test.com", password_hash=hash_password("alice-pass-1"), role="client")
        self.bob = Person(name="Bob", email="bob@test.com", password_hash=hash_password("bob-pass-1"), role="client")
        db.add_all([self.admin, self.alice, self.bob])
        db.commit()

        video_path = os.path.join(self.tmpdir.name, "run.mp4")
        vertical_path = os.path.join(self.tmpdir.name, "run_vertical.mp4")
        for path in (video_path, vertical_path):
            with open(path, "wb") as handle:
                handle.write(b"fake-mp4")
        self.run = VideoAnalysis(
            person_id=self.alice.id,
            attempt_number=1,
            video_name="uuid_internal_name.mp4",
            output_video_path=video_path,
            display_mode="athlete",
            video_exports={"athlete": {"vertical": {"path": vertical_path}}},
            input_video_path="temp_videos/secret_input.mp4",
            s3_video_key=None,
            pressure_score=150,
            balance_score=160,
            rotation_score=140,
            edging_score=130,
            instructor_notes="Nice work today.",
            notes_source="ai",
            status="completed",
        )
        db.add(self.run)
        db.commit()
        self.ids = {"admin": self.admin.id, "alice": self.alice.id, "bob": self.bob.id, "run": self.run.id}
        db.close()

    def tearDown(self):
        wait_for_export_threads()
        self.session_patch.stop()
        self.tmpdir.cleanup()

    def login(self, email, password):
        response = self.client.post("/api/auth/login", json={"email": email, "password": password})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["access_token"]

    def headers(self, token):
        return {"Authorization": f"Bearer {token}"}


class AccountAndSharingTests(_Fixture):
    # ── Change password ────────────────────────────────────────────────
    def test_change_password_rotates_sessions(self):
        old_token = self.login("alice@test.com", "alice-pass-1")
        wrong = self.client.post(
            "/api/auth/password/change",
            json={"current_password": "nope", "new_password": "new-pass-123"},
            headers=self.headers(old_token),
        )
        self.assertEqual(wrong.status_code, 400)

        changed = self.client.post(
            "/api/auth/password/change",
            json={"current_password": "alice-pass-1", "new_password": "new-pass-123"},
            headers=self.headers(old_token),
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        new_token = changed.json()["access_token"]

        self.assertEqual(self.client.get("/api/auth/me", headers=self.headers(old_token)).status_code, 401)
        self.assertEqual(self.client.get("/api/auth/me", headers=self.headers(new_token)).status_code, 200)
        self.login("alice@test.com", "new-pass-123")

    # ── Forgot / reset password ────────────────────────────────────────
    def test_forgot_password_unavailable_without_smtp(self):
        with mock.patch("services.email_service.is_configured", return_value=False):
            response = self.client.post("/api/auth/password/forgot", json={"email": "alice@test.com"})
        self.assertEqual(response.status_code, 503)

    def test_reset_password_with_emailed_code(self):
        sent = {}

        def capture(email, code, purpose, minutes):
            sent["code"] = code

        with mock.patch("services.email_service.is_configured", return_value=True), \
                mock.patch("services.email_service.send_code_email", side_effect=capture):
            unknown = self.client.post("/api/auth/password/forgot", json={"email": "nobody@test.com"})
            known = self.client.post("/api/auth/password/forgot", json={"email": "alice@test.com"})
            self.assertEqual(unknown.json(), known.json())  # no account enumeration

            bad = self.client.post(
                "/api/auth/password/reset",
                json={"email": "alice@test.com", "code": "000000" if sent["code"] != "000000" else "111111", "new_password": "reset-pass-1"},
            )
            self.assertEqual(bad.status_code, 400)

            good = self.client.post(
                "/api/auth/password/reset",
                json={"email": "alice@test.com", "code": sent["code"], "new_password": "reset-pass-1"},
            )
            self.assertEqual(good.status_code, 200, good.text)

            reused = self.client.post(
                "/api/auth/password/reset",
                json={"email": "alice@test.com", "code": sent["code"], "new_password": "another-pass-1"},
            )
            self.assertEqual(reused.status_code, 400)
        self.login("alice@test.com", "reset-pass-1")

    def test_reset_code_locks_after_too_many_guesses(self):
        sent = {}
        with mock.patch("services.email_service.is_configured", return_value=True), \
                mock.patch("services.email_service.send_code_email", side_effect=lambda e, c, p, m: sent.update(code=c)):
            self.client.post("/api/auth/password/forgot", json={"email": "alice@test.com"})
            wrong = "000000" if sent["code"] != "000000" else "111111"
            for _ in range(5):
                self.client.post("/api/auth/password/reset", json={"email": "alice@test.com", "code": wrong, "new_password": "reset-pass-1"})
            locked = self.client.post(
                "/api/auth/password/reset",
                json={"email": "alice@test.com", "code": sent["code"], "new_password": "reset-pass-1"},
            )
        self.assertEqual(locked.status_code, 400)

    def test_admin_reset_needs_no_email_and_only_resets_the_admin(self):
        sent = []
        with mock.patch.dict(os.environ, {"DEFAULT_ADMIN_EMAIL": "admin@test.com"}),                 mock.patch("services.email_service.is_configured", return_value=True),                 mock.patch("services.email_service.send_code_email", side_effect=lambda e, c, p, m: sent.append((e, c))):
            # The client page never sends a code for an admin account.
            self.client.post("/api/auth/password/forgot", json={"email": "admin@test.com"})
            self.assertEqual(sent, [])

            requested = self.client.post("/api/auth/password/admin-forgot")
            self.assertEqual(requested.status_code, 200, requested.text)
            self.assertIn("a•••n@test.com", requested.json()["message"])
            self.assertEqual(sent[0][0], "admin@test.com")
            code = sent[0][1]

            # An admin code can't be used through the client reset.
            wrong_page = self.client.post("/api/auth/password/reset",
                                          json={"email": "admin@test.com", "code": code, "new_password": "new-admin-pass"})
            self.assertEqual(wrong_page.status_code, 400)
            reset = self.client.post("/api/auth/password/admin-reset", json={"code": code, "new_password": "new-admin-pass"})
            self.assertEqual(reset.status_code, 200, reset.text)
        self.login("admin@test.com", "new-admin-pass")
        self.login("alice@test.com", "alice-pass-1")  # clients untouched

    def test_admin_reset_unavailable_without_configured_admin(self):
        with mock.patch.dict(os.environ, {"DEFAULT_ADMIN_EMAIL": "alice@test.com"}),                 mock.patch("services.email_service.is_configured", return_value=True):
            self.assertEqual(self.client.post("/api/auth/password/admin-forgot").status_code, 503)

    # ── Signup verification ────────────────────────────────────────────
    def test_signup_requires_code_when_email_configured(self):
        sent = {}
        details = {"name": "Carol", "email": "carol@test.com", "phone": None, "password": "carol-pass-1"}
        with mock.patch("services.email_service.is_configured", return_value=True), \
                mock.patch("services.email_service.send_code_email", side_effect=lambda e, c, p, m: sent.update(code=c)):
            self.assertEqual(self.client.post("/api/auth/signup", json=details).status_code, 400)
            self.assertEqual(self.client.post("/api/auth/signup/request-code", json=details).status_code, 200)
            self.assertEqual(self.client.post("/api/auth/login", json={"email": "carol@test.com", "password": "carol-pass-1"}).status_code, 401)
            verified = self.client.post("/api/auth/signup/verify", json={"email": "carol@test.com", "code": sent["code"]})
        self.assertEqual(verified.status_code, 200, verified.text)
        self.assertEqual(verified.json()["user"]["role"], "client")
        self.login("carol@test.com", "carol-pass-1")

    def test_direct_signup_still_works_without_smtp(self):
        with mock.patch("services.email_service.is_configured", return_value=False):
            response = self.client.post(
                "/api/auth/signup",
                json={"name": "Dan", "email": "dan@test.com", "password": "dan-pass-12"},
            )
        self.assertEqual(response.status_code, 200, response.text)

    # ── Downloads ──────────────────────────────────────────────────────
    def test_download_requires_login_and_ownership(self):
        run_id = self.ids["run"]
        self.assertEqual(self.client.get(f"/api/videos/{run_id}/download-url").status_code, 401)
        self.assertEqual(self.client.get(f"/api/videos/{run_id}/download").status_code, 422)

        bob = self.login("bob@test.com", "bob-pass-1")
        self.assertEqual(self.client.get(f"/api/videos/{run_id}/download-url", headers=self.headers(bob)).status_code, 404)

        alice = self.login("alice@test.com", "alice-pass-1")
        link = self.client.get(f"/api/videos/{run_id}/download-url?format=vertical", headers=self.headers(alice))
        self.assertEqual(link.status_code, 200, link.text)
        self.assertTrue(link.json()["filename"].endswith("_vertical.mp4"))
        file_response = self.client.get(link.json()["download_url"])
        self.assertEqual(file_response.status_code, 200)
        self.assertEqual(file_response.content, b"fake-mp4")

        self.assertTrue(link.json()["filename"].endswith("_athlete_vertical.mp4"))

        missing = self.client.get(f"/api/videos/{run_id}/download-url?format=square", headers=self.headers(alice))
        self.assertEqual(missing.status_code, 404)
        coach = self.client.get(f"/api/videos/{run_id}/download-url?mode=coach&format=vertical", headers=self.headers(alice))
        self.assertEqual(coach.status_code, 404)
        bad_mode = self.client.get(f"/api/videos/{run_id}/download-url?mode=director", headers=self.headers(alice))
        self.assertEqual(bad_mode.status_code, 422)

        admin = self.login("admin@test.com", "admin-pass-1")
        self.assertEqual(self.client.get(f"/api/videos/{run_id}/download-url", headers=self.headers(admin)).status_code, 200)

    def test_download_token_cannot_be_used_as_session(self):
        alice = self.login("alice@test.com", "alice-pass-1")
        link = self.client.get(f"/api/videos/{self.ids['run']}/download-url", headers=self.headers(alice)).json()
        download_token = link["download_url"].split("token=")[1]
        self.assertEqual(self.client.get("/api/auth/me", headers=self.headers(download_token)).status_code, 401)
        wrong_format = link["download_url"].replace("format=original", "format=vertical")
        self.assertEqual(self.client.get(wrong_format).status_code, 403)
        wrong_mode = link["download_url"].replace("mode=athlete", "mode=coach")
        self.assertEqual(self.client.get(wrong_mode).status_code, 403)

    def test_pending_versions_report_preparing(self):
        from datetime import datetime as dt

        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update(
            {"exports_status": "processing", "exports_updated_at": dt.utcnow(), "export_requests": ["coach:square"]}
        )
        db.commit()
        db.close()
        alice = self.login("alice@test.com", "alice-pass-1")
        response = self.client.get(f"/api/videos/{self.ids['run']}/download-url?mode=coach&format=square", headers=self.headers(alice))
        self.assertEqual(response.status_code, 409)
        row = self.client.get("/api/me/attempts", headers=self.headers(alice)).json()[0]
        self.assertTrue(row["downloads_pending"])

    # ── Client data visibility ─────────────────────────────────────────
    def test_client_attempts_hide_internal_fields(self):
        alice = self.login("alice@test.com", "alice-pass-1")
        rows = self.client.get("/api/me/attempts", headers=self.headers(alice)).json()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for hidden in ("input_video_path", "output_video_path", "s3_video_key", "video_name", "notes_source", "is_archived", "display_mode"):
            self.assertNotIn(hidden, row)
        self.assertEqual(row["instructor_notes"], "Nice work today.")
        self.assertEqual(row["available_downloads"], {"athlete": ["original", "vertical"], "coach": []})
        self.assertFalse(row["downloads_pending"])

    def test_client_cannot_open_archived_run(self):
        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update({"is_archived": True})
        db.commit()
        db.close()
        alice = self.login("alice@test.com", "alice-pass-1")
        self.assertEqual(self.client.get(f"/api/me/attempts/{self.ids['run']}", headers=self.headers(alice)).status_code, 404)
        self.assertEqual(self.client.get(f"/api/videos/{self.ids['run']}/download-url", headers=self.headers(alice)).status_code, 404)


class CoachEditTests(_Fixture):
    def test_coach_edit_marks_source_and_keeps_pb_context(self):
        from routes import admin_portal

        local_app = FastAPI()
        local_app.include_router(admin_portal.router, prefix="/api")
        local_app.dependency_overrides[database.get_db] = override_get_db
        client = TestClient(local_app)

        # An earlier, better run so the edited run has personal bests to compare against.
        db = TestingSession()
        earlier = VideoAnalysis(
            person_id=self.ids["alice"], attempt_number=0, video_name="earlier.mp4",
            pressure_score=200, balance_score=200, rotation_score=200, edging_score=200,
            status="completed", created_at=datetime(2020, 1, 1),
        )
        db.add(earlier)
        db.commit()
        db.close()

        captured = {}

        def fake_report(result, timeline, context=None):
            captured.update(context or {})
            return {"report_path": None}

        admin = self.login("admin@test.com", "admin-pass-1")
        with mock.patch("services.report_generator.generate_basic_report", side_effect=fake_report), \
                mock.patch.object(admin_portal.S3Manager, "is_enabled", return_value=False):
            response = client.patch(
                f"/api/admin/attempts/{self.ids['run']}/notes",
                json={"notes": "  Lovely rhythm today, keep it up.  "},
                headers=self.headers(admin),
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["instructor_notes"], "Lovely rhythm today, keep it up.")
        self.assertEqual(body["notes_source"], "coach")
        self.assertEqual(captured["instructor_notes"], "Lovely rhythm today, keep it up.")
        self.assertIn("blue_iq", captured["previous_personal_bests"])
        self.assertEqual(captured["previous_personal_bests"]["pressure"]["score"], 200)
        self.assertEqual(captured["first_run_blue_iq"], 200)


class ExportPoolTests(_Fixture):
    def test_background_exports_record_every_version_and_clean_up(self):
        import threading
        from services import video_exports

        input_copy = os.path.join(self.tmpdir.name, "input.mp4")
        overlay = os.path.join(self.tmpdir.name, "layer.avi")
        for path in (input_copy, overlay):
            with open(path, "wb") as handle:
                handle.write(b"x")

        def fake_render(manifest, mode, fmt, progress=None):
            path = os.path.join(self.tmpdir.name, f"{mode}_{fmt}.mp4")
            with open(path, "wb") as handle:
                handle.write(b"video")
            return path

        manifest = {"display_mode": "athlete", "overlay_path": overlay}
        done = threading.Event()
        original_finalize = video_exports._finalize

        def finalize_and_signal(*args):
            original_finalize(*args)
            done.set()

        with mock.patch("database.SessionLocal", TestingSession), \
                mock.patch.object(video_exports, "render_variant", side_effect=fake_render), \
                mock.patch.object(video_exports, "_finalize", side_effect=finalize_and_signal), \
                mock.patch("services.aws_s3.S3Manager.is_enabled", return_value=False):
            video_exports.submit_exports(self.ids["run"], manifest, cleanup_paths=[input_copy])
            self.assertTrue(done.wait(30))

        db = TestingSession()
        run = db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).first()
        self.assertEqual(run.exports_status, "ready")
        self.assertEqual(sorted(run.video_exports["athlete"]), ["square", "vertical"])
        self.assertEqual(sorted(run.video_exports["coach"]), ["original", "square", "vertical"])
        self.assertEqual(run.available_downloads["coach"], ["original", "vertical", "square"])
        db.close()
        self.assertFalse(os.path.exists(input_copy))
        self.assertFalse(os.path.exists(overlay))


class AdminRunFilterTests(_Fixture):
    def test_filter_runs_by_day_and_athlete_name_or_email(self):
        from routes import admin_portal

        local_app = FastAPI()
        local_app.include_router(admin_portal.router, prefix="/api")
        local_app.dependency_overrides[database.get_db] = override_get_db
        client = TestClient(local_app)

        db = TestingSession()
        db.add(VideoAnalysis(
            person_id=self.ids["bob"], attempt_number=1, video_name="bob.mp4",
            pressure_score=100, balance_score=100, rotation_score=100, edging_score=100,
            status="completed", created_at=datetime(2026, 10, 5, 15, 0),
        ))
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update(
            {"created_at": datetime(2026, 10, 13, 9, 30)}
        )
        db.commit()
        db.close()

        admin = self.headers(self.login("admin@test.com", "admin-pass-1"))

        def names(**params):
            response = client.get("/api/admin/attempts", params=params, headers=admin)
            self.assertEqual(response.status_code, 200, response.text)
            return sorted(row["client_name"] for row in response.json())

        self.assertEqual(names(), ["Alice", "Bob"])
        # A local day sent as an aware UTC range (e.g. UTC+5 midnight to midnight).
        self.assertEqual(names(start="2026-10-12T19:00:00Z", end="2026-10-13T19:00:00Z"), ["Alice"])
        self.assertEqual(names(start="2026-10-05T00:00:00", end="2026-10-06T00:00:00"), ["Bob"])
        self.assertEqual(names(start="2026-10-14T00:00:00", end="2026-10-15T00:00:00"), [])
        self.assertEqual(names(q="ali"), ["Alice"])
        self.assertEqual(names(q="BOB@TEST"), ["Bob"])
        self.assertEqual(names(q="%"), [])
        self.assertEqual(names(q="bob", start="2026-10-12T19:00:00Z", end="2026-10-13T19:00:00Z"), [])
        self.assertEqual(names(person_ids=str(self.ids["bob"])), ["Bob"])
        self.assertEqual(names(person_ids=f'{self.ids["alice"]},{self.ids["bob"]}'), ["Alice", "Bob"])
        self.assertEqual(names(person_ids=""), [])
        response = client.get("/api/admin/attempts", params={"person_ids": "1,x"}, headers=admin)
        self.assertEqual(response.status_code, 422)

        # Athlete list counts all of an athlete's runs, not just the loaded day.
        users = {row["name"]: row for row in client.get("/api/admin/users", headers=admin).json()}
        self.assertEqual(users["Alice"]["run_count"], 1)
        self.assertEqual(users["Bob"]["run_count"], 1)
        self.assertEqual(users["Admin"]["run_count"], 0)


class RunHistoryTests(_Fixture):
    def test_run_history_compares_latest_run_with_first_run(self):
        from routes import admin_portal

        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update(
            {"created_at": datetime(2026, 10, 1, 9, 0)}
        )
        db.add(VideoAnalysis(
            person_id=self.ids["alice"], attempt_number=2, video_name="second.mp4",
            pressure_score=170, balance_score=150, rotation_score=160, edging_score=150,
            status="completed", created_at=datetime(2026, 10, 8, 9, 0),
        ))
        db.commit()
        db.close()

        alice = self.headers(self.login("alice@test.com", "alice-pass-1"))
        mine = self.client.get("/api/me/trends", headers=alice).json()
        self.assertEqual(mine["first_run"]["scores"]["pressure"], 150)
        self.assertEqual(mine["first_run"]["session_number"], 1)
        self.assertEqual(mine["current_vs_first_run"]["pressure"]["difference"], 20)
        self.assertEqual(mine["current_vs_first_run"]["balance"]["difference"], -10)
        self.assertFalse(mine["current_vs_first_run"]["blue_iq"]["is_first_run"])

        local_app = FastAPI()
        local_app.include_router(admin_portal.router, prefix="/api")
        local_app.dependency_overrides[database.get_db] = override_get_db
        admin = self.headers(self.login("admin@test.com", "admin-pass-1"))
        theirs = TestClient(local_app).get(f"/api/admin/users/{self.ids['alice']}/trends", headers=admin).json()
        self.assertEqual(theirs["current_vs_first_run"], mine["current_vs_first_run"])
        self.assertEqual(TestClient(local_app).get("/api/admin/users/9999/trends", headers=admin).status_code, 404)


class ClientRunNumberingTests(_Fixture):
    def test_client_sees_own_sequence_and_full_history(self):
        from routes import admin_portal

        db = TestingSession()
        db.query(VideoAnalysis).filter(VideoAnalysis.id == self.ids["run"]).update(
            {"attempt_number": 49, "created_at": datetime(2026, 9, 1, 9, 0)}
        )
        for day in range(20):
            db.add(VideoAnalysis(
                person_id=self.ids["alice"], attempt_number=50 + day * 2, video_name=f"r{day}.mp4",
                pressure_score=150, balance_score=150, rotation_score=150, edging_score=150,
                status="completed", created_at=datetime(2026, 9, 2 + day, 9, 0),
            ))
        db.commit()
        db.close()

        alice = self.headers(self.login("alice@test.com", "alice-pass-1"))
        rows = self.client.get("/api/me/attempts", headers=alice).json()
        self.assertEqual(sorted(row["run_number"] for row in rows), list(range(1, 22)))
        self.assertNotIn("attempt_number", rows[0])

        trends = self.client.get("/api/me/trends", headers=alice).json()
        self.assertEqual(trends["time_series"]["runs"], list(range(1, 22)))  # all runs, not 15
        self.assertEqual(trends["first_run"]["run_number"], 1)

        bests = self.client.get("/api/me/personal-bests", headers=alice).json()["personal_bests"]
        self.assertLessEqual(bests["blue_iq"]["run_number"], 21)

        link = self.client.get(f"/api/videos/{self.ids['run']}/download-url", headers=alice).json()
        self.assertEqual(link["filename"], "bluerun_run_1_athlete_original.mp4")

        local_app = FastAPI()
        local_app.include_router(admin_portal.router, prefix="/api")
        local_app.dependency_overrides[database.get_db] = override_get_db
        admin = self.headers(self.login("admin@test.com", "admin-pass-1"))
        admin_trends = TestClient(local_app).get(f"/api/admin/users/{self.ids['alice']}/trends", headers=admin).json()
        self.assertEqual(admin_trends["time_series"]["runs"][:2], [49, 50])


class ApiCompressionTests(unittest.TestCase):
    def test_api_json_is_gzipped_but_downloads_are_not(self):
        from fastapi.responses import Response
        from services.api_gzip import ApiGZipMiddleware

        local_app = FastAPI()
        local_app.add_middleware(ApiGZipMiddleware, minimum_size=1000)

        @local_app.get("/api/rows")
        def rows():
            return [{"name": "Alice", "score": i} for i in range(200)]

        @local_app.get("/api/videos/1/download")
        def download():
            return Response(b"x" * 5000, media_type="video/mp4")

        @local_app.get("/outputs/run.mp4")
        def static_file():
            return Response(b"x" * 5000, media_type="video/mp4")

        client = TestClient(local_app)
        gzip_header = {"Accept-Encoding": "gzip"}
        self.assertEqual(client.get("/api/rows", headers=gzip_header).headers.get("content-encoding"), "gzip")
        self.assertIsNone(client.get("/api/videos/1/download", headers=gzip_header).headers.get("content-encoding"))
        self.assertIsNone(client.get("/outputs/run.mp4", headers=gzip_header).headers.get("content-encoding"))


class ProgressLineTests(unittest.TestCase):
    def line(self, current, first, session=6):
        from services.report_generator import _progress_line

        return _progress_line({
            "final_scores": {"blue_iq": current},
            "session_number": session,
            "attempt_number": 3,
            "first_run_blue_iq": first,
        })

    def test_returning_client_sees_gain_since_first_run(self):
        self.assertEqual(self.line(168, 145), ("SESSION 6", "Up 23 points since your first run"))
        self.assertEqual(self.line(146, 145)[1], "Up 1 point since your first run")

    def test_first_run_gets_welcome_line(self):
        from services.report_generator import FIRST_RUN_LINES

        self.assertIn(self.line(150, None, session=1)[1], FIRST_RUN_LINES)

    def test_lower_score_never_mentions_the_drop(self):
        from services.report_generator import ENCOURAGEMENT_LINES

        message = self.line(150, 190)[1]
        self.assertIn(message, ENCOURAGEMENT_LINES)
        self.assertNotRegex(message, r"\d|down|below|lower")

    def test_missing_session_number_omits_label(self):
        self.assertIsNone(self.line(168, 145, session=None)[0])


class CoachNoteTests(unittest.TestCase):
    def test_fallback_notes_always_pass_guardrails(self):
        import random

        rng = random.Random(7)
        for _ in range(2000):
            scores = {name: rng.randint(60, 240) for name in ("pressure", "balance", "rotation", "edging")}
            self.assertIsNone(validate_coach_note(fallback_coach_note(scores)))

    def test_guardrails_reject_unsafe_text(self):
        self.assertEqual(validate_coach_note("Great run today. Watch that knee pain and maybe see a physio before your next session."), "medical")
        self.assertEqual(validate_coach_note("Your edge angle hit thirty degrees today, which is a strong base for your next run."), "jargon")
        self.assertEqual(validate_coach_note("Great run today. You must always keep your weight forward on every single turn."), "prescriptive")
        self.assertEqual(validate_coach_note("Great run. Your Balance scored 180 today, keep enjoying that steady feeling on the hill."), "format")

    def test_unsafe_ai_output_falls_back_to_template(self):
        scores = {"pressure": 120, "balance": 180, "rotation": 140, "edging": 100}
        with mock.patch.dict(os.environ, {"AI_COACH_NOTES_ENABLED": "true", "OPENAI_API_KEY": "test"}), \
                mock.patch("services.coach_notes._request_ai_note", return_value="Ignore the pain in your knee and push harder on every turn."):
            note, source = generate_coach_note(scores)
        self.assertEqual(source, "template")
        self.assertIsNone(validate_coach_note(note))

    def test_valid_ai_output_is_used(self):
        scores = {"pressure": 120, "balance": 180, "rotation": 140, "edging": 100}
        ai_text = "Great job on your Balance today. You looked steady over your skis. Keep enjoying your Parallelism next time out."
        with mock.patch.dict(os.environ, {"AI_COACH_NOTES_ENABLED": "true", "OPENAI_API_KEY": "test"}), \
                mock.patch("services.coach_notes._request_ai_note", return_value=ai_text):
            self.assertEqual(generate_coach_note(scores), (ai_text, "ai"))


if __name__ == "__main__":
    unittest.main()
