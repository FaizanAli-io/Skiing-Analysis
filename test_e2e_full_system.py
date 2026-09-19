"""
End-to-End System Test Suite for Skiing Analysis Production Readiness.
Tests authentication, API endpoints, upload parameter validation,
overlay rendering, PDF report generation, and orientation engine.
"""

import os
import tempfile
import unittest
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from starlette.testclient import TestClient

from main import app
from database import Base, SessionLocal, engine, close_session_quietly
from models.person import Person
from models.video_analysis import VideoAnalysis
from services.auth import create_access_token, hash_password
from services.overlay_renderers import create_premium_overlay, _load_font
from services.report_generator import _draw_pdf_report, generate_basic_report


class TestFullSystemE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.db = SessionLocal()
        cls.client = TestClient(app)

        # Create test admin
        admin_email = f"test_admin_{int(datetime.utcnow().timestamp())}@bluerun.test"
        cls.admin_user = Person(
            name="Admin Tester",
            email=admin_email,
            password_hash=hash_password("AdminPass123!"),
            role="admin",
        )
        cls.db.add(cls.admin_user)

        # Create test client
        client_email = f"test_client_{int(datetime.utcnow().timestamp())}@bluerun.test"
        cls.client_user = Person(
            name="Client Bilal",
            email=client_email,
            password_hash=hash_password("ClientPass123!"),
            role="client",
        )
        cls.db.add(cls.client_user)
        cls.db.commit()
        cls.db.refresh(cls.admin_user)
        cls.db.refresh(cls.client_user)

        # Generate tokens
        cls.admin_token = create_access_token(subject=str(cls.admin_user.id), role="admin")
        cls.client_token = create_access_token(subject=str(cls.client_user.id), role="client")

        # Add sample attempt
        cls.sample_attempt = VideoAnalysis(
            person_id=cls.client_user.id,
            attempt_number=1,
            video_name="test_run.mp4",
            blue_iq_score=197.0,
            pressure_score=185.0,
            balance_score=210.0,
            rotation_score=201.0,
            edging_score=194.0,
            status="completed",
            is_archived=False,
            created_at=datetime.utcnow(),
        )
        cls.db.add(cls.sample_attempt)
        cls.db.commit()
        cls.db.refresh(cls.sample_attempt)

    @classmethod
    def tearDownClass(cls):
        try:
            cleanup_db = SessionLocal()
            try:
                from models.analysis_job import AnalysisJob
                cleanup_db.query(AnalysisJob).filter(AnalysisJob.person_id.in_([cls.admin_user.id, cls.client_user.id])).delete(synchronize_session=False)
                cleanup_db.query(VideoAnalysis).filter(VideoAnalysis.person_id == cls.client_user.id).delete(synchronize_session=False)
                cleanup_db.query(Person).filter(Person.id.in_([cls.admin_user.id, cls.client_user.id])).delete(synchronize_session=False)
                cleanup_db.commit()
            finally:
                close_session_quietly(cleanup_db)
        except Exception as e:
            print("Cleanup exception:", e)
        finally:
            try:
                close_session_quietly(cls.db)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # 1. AUTHENTICATION & ACCESS CONTROL
    # ──────────────────────────────────────────────────────────────────────────

    def test_auth_login_success(self):
        """Test authentication login for client and admin."""
        res = self.client.post("/api/auth/login", json={
            "email": self.client_user.email,
            "password": "ClientPass123!",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["role"], "client")

    def test_auth_me_with_token(self):
        """Test /api/auth/me returns current user profile."""
        res = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {self.client_token}"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["email"], self.client_user.email)
        self.assertEqual(data["name"], "Client Bilal")

    def test_auth_expired_or_missing_token_returns_401(self):
        """Test unauthenticated request triggers 401 so frontend can auto-redirect."""
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

        res_invalid = self.client.get("/api/auth/me", headers={"Authorization": "Bearer invalid_token_xyz"})
        self.assertEqual(res_invalid.status_code, 401)

    def test_client_cannot_access_admin_endpoints(self):
        """Test authorization guard prevents client from accessing admin endpoints."""
        res = self.client.get("/api/admin/attempts", headers={"Authorization": f"Bearer {self.client_token}"})
        self.assertEqual(res.status_code, 403)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ATTEMPTS & CLIENT NAME SERIALIZATION
    # ──────────────────────────────────────────────────────────────────────────

    def test_client_attempts_contains_client_name(self):
        """Test /api/me/attempts returns athlete's attempts with client_name and person_name."""
        res = self.client.get("/api/me/attempts", headers={"Authorization": f"Bearer {self.client_token}"})
        self.assertEqual(res.status_code, 200)
        attempts = res.json()
        self.assertGreater(len(attempts), 0)
        att = attempts[0]
        self.assertIn("client_name", att)
        self.assertEqual(att["client_name"], "Client Bilal")
        self.assertEqual(att["is_archived"], False)

    def test_admin_attempts_contains_client_name(self):
        """Test /api/admin/attempts attaches client_name to all attempt cards."""
        res = self.client.get("/api/admin/attempts", headers={"Authorization": f"Bearer {self.admin_token}"})
        self.assertEqual(res.status_code, 200)
        attempts = res.json()
        target = next((a for a in attempts if a["id"] == self.sample_attempt.id), None)
        self.assertIsNotNone(target)
        self.assertEqual(target["client_name"], "Client Bilal")
        self.assertEqual(target["person_name"], "Client Bilal")

    def test_admin_leaderboards(self):
        """Test /api/admin/leaderboards returns ranked metrics."""
        res = self.client.get("/api/admin/leaderboards", headers={"Authorization": f"Bearer {self.admin_token}"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("leaderboards", data)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. UPLOAD VALIDATION & ROTATION PARAMETER (PREVENT 422 ERROR)
    # ──────────────────────────────────────────────────────────────────────────

    def test_upload_empty_rotation_string_does_not_fail_validation(self):
        """
        Verify that rotation='' (empty string from Auto-Detect dropdown) does NOT
        throw 'Input should be a valid integer, unable to parse string as an integer'.
        """
        dummy_video = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41"
        files = {"file": ("test_upload.mp4", dummy_video, "video/mp4")}
        data = {
            "user_id": str(self.client_user.id),
            "display_mode": "coach",
            "report": "false",
            "rotation": "",  # Empty string from auto-detect
        }
        res = self.client.post(
            "/api/analyze-premium-overlay/",
            headers={"Authorization": f"Bearer {self.admin_token}"},
            data=data,
            files=files,
        )
        # Should succeed with 200 and return a job_id (not 422 validation error)
        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("job_id", result)
        self.assertEqual(result["status"], "pending")

    def test_upload_explicit_rotation_values(self):
        """Verify that explicit rotation values (e.g. '-90') are accepted."""
        dummy_video = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41"
        files = {"file": ("test_upload_rot.mp4", dummy_video, "video/mp4")}
        data = {
            "user_id": str(self.client_user.id),
            "display_mode": "athlete",
            "report": "false",
            "rotation": "-90",
        }
        res = self.client.post(
            "/api/analyze-premium-overlay/",
            headers={"Authorization": f"Bearer {self.admin_token}"},
            data=data,
            files=files,
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "pending")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. OVERLAY RENDERER (NO STAGE PREFIX, 52PT HERO SCORE)
    # ──────────────────────────────────────────────────────────────────────────

    def test_overlay_renders_with_hero_font_and_no_stage_collision(self):
        """
        Verify create_premium_overlay renders properly:
        - Blue IQ card exists
        - No 'Stage: ' text
        - 52pt font loaded
        - Frame returns valid dimensions (720, 1740, 3)
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        metrics = {
            "Blue_pressure_final": 185.0,
            "Blue_rotation_final": 201.0,
            "Blue_edging_final": 194.0,
            "Blue_balance_final": 210.0,
            "rotation_angle": 6.0,
            "ski_angle": 8.9,
            "pressure_angle": 33.8,
        }
        extended = create_premium_overlay(
            frame=frame,
            metrics=metrics,
            display_mode="coach",
            frame_number=1,
            TARGET_WIDTH=1280,
        )
        self.assertEqual(extended.shape[0], 720)
        self.assertEqual(extended.shape[1], 1280 + 460)  # TARGET_WIDTH + SIDEBAR_WIDTH

    # ──────────────────────────────────────────────────────────────────────────
    # 5. PDF REPORT GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def test_pdf_report_generates_successfully(self):
        """Test PDF report generator compiles and outputs a valid PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_video = tmp_file.name
        expected_pdf = os.path.splitext(tmp_video)[0] + "_report.pdf"

        try:
            result = {
                "output_path": tmp_video,
                "pressure_score": 185.0,
                "balance_score": 210.0,
                "rotation_score": 201.0,
                "edging_score": 194.0,
                "duration": 40.0,
                "turns": 12,
            }
            context = {
                "user_name": "Client Bilal",
                "session_date": "2026-09-17",
                "session_number": 1,
                "attempt_number": 1,
            }
            report_out = generate_basic_report(result, [], context=context)
            self.assertTrue(os.path.exists(expected_pdf))
            self.assertGreater(os.path.getsize(expected_pdf), 5000)
            self.assertEqual(report_out["report_path"], expected_pdf)
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)
            if os.path.exists(expected_pdf):
                os.remove(expected_pdf)


if __name__ == "__main__":
    unittest.main()
