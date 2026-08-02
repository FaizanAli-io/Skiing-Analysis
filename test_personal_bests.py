import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace

from models.person import Person
from models.video_analysis import VideoAnalysis
from services.personal_bests import (
    build_leaderboards,
    build_personal_best_history,
    compare_current_scores_to_previous,
    display_score,
)
from services.report_generator import generate_basic_report


def attempt(
    analysis_id,
    person_id,
    run_number,
    created_at,
    blue_iq,
    pressure,
    balance,
    rotation,
    edging,
    status="completed",
):
    return SimpleNamespace(
        id=analysis_id,
        person_id=person_id,
        attempt_number=run_number,
        created_at=created_at,
        timestamp=created_at,
        status=status,
        blue_iq_score=blue_iq,
        pressure_score=pressure,
        balance_score=balance,
        rotation_score=rotation,
        edging_score=edging,
    )


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self.rows)


class FakeDatabase:
    def __init__(self, attempts, people):
        self.attempts = attempts
        self.people = people

    def query(self, model):
        if model is VideoAnalysis:
            return FakeQuery(self.attempts)
        if model is Person:
            return FakeQuery(self.people)
        raise AssertionError(f"Unexpected query model: {model}")


class PersonalBestTests(unittest.TestCase):
    def setUp(self):
        self.first = attempt(
            1,
            7,
            1,
            datetime(2026, 7, 1, 10, 0),
            150.1,
            150.1,
            140.2,
            160.0,
            170.4,
        )
        self.second = attempt(
            2,
            7,
            2,
            datetime(2026, 7, 1, 11, 0),
            166.1,
            164.1,
            170.1,
            165.5,
            175.1,
        )
        self.third = attempt(
            3,
            7,
            3,
            datetime(2026, 7, 8, 9, 0),
            160.0,
            160.0,
            169.5,
            162.0,
            174.5,
        )

    def test_display_scores_use_ceiling(self):
        self.assertEqual(display_score(150.01), 151)
        self.assertEqual(display_score(240.5), 240)
        self.assertIsNone(display_score(None))

    def test_history_tracks_context_and_strict_records(self):
        history = build_personal_best_history([
            self.third,
            self.first,
            self.second,
        ])

        pressure_best = history["personal_bests"]["pressure"]
        self.assertEqual(pressure_best["score"], 165)
        self.assertEqual(pressure_best["analysis_id"], 2)
        self.assertEqual(pressure_best["session_number"], 1)
        self.assertEqual(pressure_best["run_number"], 2)
        self.assertEqual(pressure_best["date"], "2026-07-01")

        first_pressure = history["comparisons_by_attempt"][1]["pressure"]
        self.assertEqual(first_pressure["status"], "baseline")
        self.assertEqual(first_pressure["personal_best_score"], 151)

        second_pressure = history["comparisons_by_attempt"][2]["pressure"]
        self.assertEqual(second_pressure["status"], "new_personal_best")
        self.assertEqual(second_pressure["personal_best_score"], 165)
        self.assertEqual(second_pressure["previous_best"]["score"], 151)
        self.assertEqual(second_pressure["difference"], 0)

        third_pressure = history["comparisons_by_attempt"][3]["pressure"]
        self.assertEqual(third_pressure["status"], "below_personal_best")
        self.assertEqual(third_pressure["points_below"], 5)
        self.assertEqual(
            history["metadata_by_attempt"][3]["session_number"],
            2,
        )

    def test_equal_ceiled_score_does_not_create_another_record(self):
        equal = attempt(
            4,
            7,
            4,
            datetime(2026, 7, 9, 9, 0),
            160,
            164.9,
            150,
            150,
            150,
        )
        history = build_personal_best_history([self.first, self.second, equal])

        self.assertNotIn("pressure", history["new_personal_bests_by_attempt"].get(4, []))
        self.assertEqual(
            history["comparisons_by_attempt"][4]["pressure"]["status"],
            "matches_personal_best",
        )

    def test_failed_attempts_are_ignored(self):
        failed = attempt(
            5,
            7,
            5,
            datetime(2026, 7, 10, 9, 0),
            240,
            240,
            240,
            240,
            240,
            status="failed",
        )
        history = build_personal_best_history([self.first, failed])

        self.assertEqual(history["personal_bests"]["blue_iq"]["analysis_id"], 1)

    def test_report_comparison_uses_previous_record(self):
        previous = build_personal_best_history([self.first])["personal_bests"]
        comparisons = compare_current_scores_to_previous(
            {
                "blue_iq": 180,
                "pressure": 170,
                "balance": 170,
                "rotation": 170,
                "edging": 170,
            },
            previous,
            session_date="2026-07-08",
            session_number=2,
            run_number=3,
        )

        self.assertEqual(comparisons["blue_iq"]["status"], "new_personal_best")
        self.assertEqual(comparisons["blue_iq"]["previous_best"]["score"], 151)
        self.assertEqual(comparisons["blue_iq"]["personal_best"]["run_number"], 3)

    def test_leaderboards_use_one_best_result_per_client(self):
        other = attempt(
            9,
            8,
            1,
            datetime(2026, 7, 2, 12, 0),
            180,
            180,
            180,
            180,
            180,
        )
        people = [
            SimpleNamespace(id=7, name="Alex", email="alex@example.com", role="client"),
            SimpleNamespace(id=8, name="Blair", email="blair@example.com", role="client"),
            SimpleNamespace(id=99, name="Admin", email="admin@example.com", role="admin"),
        ]
        db = FakeDatabase([self.first, self.second, other], people)

        leaderboards = build_leaderboards(db)

        self.assertEqual(len(leaderboards["blue_iq"]), 2)
        self.assertEqual(leaderboards["blue_iq"][0]["athlete_name"], "Blair")
        self.assertEqual(leaderboards["blue_iq"][0]["rank"], 1)
        self.assertEqual(leaderboards["blue_iq"][1]["athlete_name"], "Alex")

    def test_pdf_report_includes_personal_best_metadata(self):
        previous = build_personal_best_history([self.first])["personal_bests"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "run_2.mp4")
            result = {
                "output_path": output_path,
                "pressure_score": 170,
                "balance_score": 180,
                "rotation_score": 190,
                "edging_score": 200,
                "duration": 20,
                "turns": 8,
            }
            report = generate_basic_report(
                result,
                [],
                use_openai=False,
                context={
                    "user_name": "Alex",
                    "attempt_number": 2,
                    "session_number": 2,
                    "session_date": "2026-07-08",
                    "previous_personal_bests": previous,
                },
            )

            self.assertTrue(os.path.exists(report["report_path"]))
            self.assertEqual(
                report["personal_best_comparisons"]["blue_iq"]["status"],
                "new_personal_best",
            )
            with open(report["report_path"], "rb") as report_file:
                self.assertEqual(report_file.read(4), b"%PDF")


if __name__ == "__main__":
    unittest.main()
