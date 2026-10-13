import os
import unittest
from datetime import date
from services.report_generator import (
    APPROVED_PILLAR_ASSESSMENTS,
    APPROVED_COACHING_CUES,
    _personal_best_context_text,
    _approved_improvement_areas,
    generate_basic_report,
)

class TestClientFeedbackFeatures(unittest.TestCase):
    def test_personal_best_compact_format(self):
        comparison = {
            "status": "below_personal_best",
            "current_score": 202,
            "personal_best_score": 231,
            "points_below": 29,
            "personal_best": {
                "date": "2026-07-23",
                "session_number": 2,
                "run_number": 54,
            },
        }
        text = _personal_best_context_text(comparison)
        self.assertEqual(text, "PB 231 (-29) · Jul 23")
        self.assertLessEqual(len(text), 25)

    def test_footer_scale_twelve_step(self):
        dummy_result = {
            "output_path": "outputs/test_run.mp4",
            "pressure_score": 210,
            "balance_score": 200,
            "rotation_score": 195,
            "edging_score": 190,
            "duration": 15,
            "turns": 6,
        }
        report_data = generate_basic_report(dummy_result, [], context={"user_name": "Lev"})
        self.assertIn("Building 1-3 (60-104)", report_data.get("score_scale", ""))
        self.assertTrue(os.path.exists(report_data["report_path"]))

    def test_improvement_areas_has_next_session_focus_and_no_duplicates(self):
        scores = {
            "pressure": 215,
            "balance": 210,
            "rotation": 205,
            "edging": 180,  # lowest
        }
        bullets = _approved_improvement_areas(scores)
        self.assertTrue(len(bullets) >= 2)
        # First bullet should be targeted next session focus for the lowest pillar (edging)
        self.assertTrue(bullets[0].startswith("Focus for next session (Parallelism):"))
        
        # Ensure bullets don't duplicate the primary coaching focus cues of the pillar cards
        edging_cue = APPROVED_COACHING_CUES["edging"]["advancing"]
        for bullet in bullets:
            self.assertNotEqual(bullet, edging_cue)

    def test_varied_pillar_text_for_mastering(self):
        # Ensure the 4 pillars do not all use the identical repetitive phrase
        summaries = [
            APPROVED_PILLAR_ASSESSMENTS[p]["mastering"]
            for p in ("pressure", "balance", "rotation", "edging")
        ]
        # None should have repetitive "is mastering, with"
        for s in summaries:
            self.assertNotIn("is mastering, with", s)
        # All 4 summaries must be distinct
        self.assertEqual(len(set(summaries)), 4)

    def test_instructor_notes_generation_when_provided(self):
        dummy_result = {
            "output_path": "outputs/test_run_notes.mp4",
            "pressure_score": 195,
            "balance_score": 190,
            "rotation_score": 185,
            "edging_score": 180,
            "duration": 20,
            "turns": 8,
            "instructor_notes": "Great edge angle work today. Let's work on earlier initiation next time.",
        }
        report_data = generate_basic_report(dummy_result, [], context={"user_name": "Tina"})
        self.assertTrue(os.path.exists(report_data["report_path"]))
        self.assertIn("Coach's Note: Great edge angle work today.", report_data["report_text"])

    def test_instructor_notes_omitted_when_none_or_empty(self):
        dummy_result = {
            "output_path": "outputs/test_run_no_notes.mp4",
            "pressure_score": 195,
            "balance_score": 190,
            "rotation_score": 185,
            "edging_score": 180,
            "duration": 20,
            "turns": 8,
            "instructor_notes": "   ",
        }
        report_data = generate_basic_report(dummy_result, [], context={"user_name": "Tina"})
        self.assertTrue(os.path.exists(report_data["report_path"]))
        self.assertNotIn("Coach's Note", report_data["report_text"])

if __name__ == "__main__":
    unittest.main()
