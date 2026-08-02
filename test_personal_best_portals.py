import unittest
from datetime import datetime
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from database import get_db
from models.person import Person
from models.video_analysis import VideoAnalysis
from routes import admin_portal, client_portal
from services.auth import get_current_user, require_admin


class FakeQuery:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def offset(self, value):
        self.rows = self.rows[value:]
        return self

    def limit(self, value):
        self.rows = self.rows[:value]
        return self

    def all(self):
        return list(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDatabase:
    def __init__(self, attempts, people):
        self.attempts = attempts
        self.people = people

    def query(self, model):
        if model is VideoAnalysis:
            return FakeQuery(self.attempts)
        if model is Person:
            return FakeQuery(self.people)
        raise AssertionError(f"Unexpected model: {model}")


def analysis(analysis_id, run, created_at, scores):
    return VideoAnalysis(
        id=analysis_id,
        person_id=2,
        attempt_number=run,
        video_name=f"run_{run}.mp4",
        video_link=None,
        output_video_path=f"outputs/run_{run}.mp4",
        report_path=f"outputs/run_{run}_report.pdf",
        display_mode="coach",
        overlay_renderer="premium",
        blue_iq_score=sum(scores) / 4,
        pressure_score=scores[0],
        balance_score=scores[1],
        rotation_score=scores[2],
        edging_score=scores[3],
        turns=10,
        duration=20.0,
        status="completed",
        timestamp=created_at,
        created_at=created_at,
        updated_at=created_at,
    )


class PersonalBestPortalTests(unittest.TestCase):
    def setUp(self):
        self.client_user = Person(
            id=2,
            name="Alex",
            email="alex@example.com",
            role="client",
            is_active=True,
        )
        self.admin_user = Person(
            id=1,
            name="Coach",
            email="coach@example.com",
            role="admin",
            is_active=True,
        )
        self.attempts = [
            analysis(1, 1, datetime(2026, 7, 1, 10, 0), [150, 160, 170, 180]),
            analysis(2, 2, datetime(2026, 7, 8, 10, 0), [190, 200, 180, 210]),
        ]
        self.db = FakeDatabase(
            self.attempts,
            [self.client_user, self.admin_user],
        )

    def test_client_portal_exposes_context_without_declining_labels(self):
        app = FastAPI()
        app.include_router(client_portal.router)
        app.dependency_overrides[get_current_user] = lambda: self.client_user
        app.dependency_overrides[get_db] = lambda: self.db

        with patch.object(client_portal.S3Manager, "is_enabled", return_value=False):
            with TestClient(app) as client:
                attempts_response = client.get("/me/attempts")
                bests_response = client.get("/me/personal-bests")
                trends_response = client.get("/me/trends")

        self.assertEqual(attempts_response.status_code, 200)
        attempts = attempts_response.json()
        latest = next(item for item in attempts if item["id"] == 2)
        self.assertEqual(latest["session_number"], 2)
        self.assertIn("blue_iq", latest["new_personal_bests"])

        self.assertEqual(bests_response.status_code, 200)
        self.assertEqual(
            bests_response.json()["personal_bests"]["edging"]["run_number"],
            2,
        )

        self.assertEqual(trends_response.status_code, 200)
        trends = trends_response.json()
        self.assertIn("current_vs_personal_best", trends)
        self.assertNotIn("trends", trends)
        self.assertNotIn("improvement_rates", trends)

    def test_admin_portal_exposes_all_five_leaderboards(self):
        app = FastAPI()
        app.include_router(admin_portal.router)
        app.dependency_overrides[require_admin] = lambda: self.admin_user
        app.dependency_overrides[get_db] = lambda: self.db

        with TestClient(app) as client:
            response = client.get("/admin/leaderboards")

        self.assertEqual(response.status_code, 200)
        leaderboards = response.json()["leaderboards"]
        self.assertEqual(
            set(leaderboards),
            {"blue_iq", "pressure", "balance", "rotation", "edging"},
        )
        self.assertEqual(leaderboards["blue_iq"][0]["athlete_name"], "Alex")
        self.assertEqual(leaderboards["blue_iq"][0]["rank"], 1)

    def test_chart_limit_does_not_limit_all_time_personal_best(self):
        historical_best = analysis(
            10,
            4,
            datetime(2026, 6, 1, 10, 0),
            [220, 220, 220, 220],
        )
        current = analysis(
            11,
            5,
            datetime(2026, 7, 1, 10, 0),
            [150, 150, 150, 150],
        )
        db = FakeDatabase(
            [historical_best, current],
            [self.client_user],
        )
        app = FastAPI()
        app.include_router(client_portal.router)
        app.dependency_overrides[get_current_user] = lambda: self.client_user
        app.dependency_overrides[get_db] = lambda: db

        with TestClient(app) as client:
            response = client.get("/me/trends?limit=1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["displayed_runs"], 1)
        self.assertEqual(payload["time_series"]["blue_iq"], [150])
        self.assertEqual(payload["personal_bests"]["blue_iq"]["score"], 220)
        self.assertEqual(
            payload["current_vs_personal_best"]["blue_iq"]["points_below"],
            70,
        )


if __name__ == "__main__":
    unittest.main()
