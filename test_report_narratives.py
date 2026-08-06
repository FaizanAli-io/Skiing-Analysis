import unittest

from services.report_generator import (
    APPROVED_COACHING_CUES,
    APPROVED_PILLAR_ASSESSMENTS,
    BASELINE_COMPARISON_PATTERN,
    _build_fixed_sections,
)


PILLARS = ("pressure", "balance", "rotation", "edging")
BAND_SCORES = {
    "emerging": 83,
    "developing": 145,
    "proficient": 185,
    "excellent": 215,
}


def report_payload(scores=None, status="baseline"):
    final_scores = {
        "blue_iq": 145,
        "pressure": 145,
        "balance": 145,
        "rotation": 145,
        "edging": 145,
    }
    if scores:
        final_scores.update(scores)

    return {
        "final_scores": final_scores,
        "blueiq_level": {
            "category": "Intermediate",
            "level": 5,
            "name": "Building Rhythm",
            "key_focus": "Build repeatable movement and control.",
        },
        "personal_best_comparisons": {
            pillar: {"status": status}
            for pillar in ("blue_iq", *PILLARS)
        },
    }


class FixedReportNarrativeTests(unittest.TestCase):
    def test_every_pillar_uses_assessment_and_cue_from_its_score_band(self):
        for pillar in PILLARS:
            for band, score in BAND_SCORES.items():
                with self.subTest(pillar=pillar, band=band):
                    sections = _build_fixed_sections(
                        report_payload({pillar: score})
                    )
                    pillar_section = sections["pillars"][pillar]

                    self.assertEqual(
                        pillar_section["summary"],
                        APPROVED_PILLAR_ASSESSMENTS[pillar][band],
                    )
                    self.assertEqual(
                        pillar_section["coaching_focus"],
                        APPROVED_COACHING_CUES[pillar][band],
                    )

    def test_baseline_copy_describes_current_scores_without_progress_claims(self):
        sections = _build_fixed_sections(report_payload({
            "blue_iq": 80,
            "pressure": 63,
            "balance": 83,
            "rotation": 91,
            "edging": 84,
        }))
        narrative = " ".join([
            sections["overall"],
            *(sections["pillars"][pillar]["summary"] for pillar in PILLARS),
        ])

        self.assertIn("establishes a Blue IQ baseline of 80/240", sections["overall"])
        self.assertIn("Balance is emerging", sections["pillars"]["balance"]["summary"])
        self.assertIn("Rotational control is emerging", sections["pillars"]["rotation"]["summary"])
        self.assertIsNone(BASELINE_COMPARISON_PATTERN.search(narrative))

    def test_new_personal_best_language_requires_recorded_comparison_status(self):
        baseline = _build_fixed_sections(report_payload({"blue_iq": 180}))
        personal_best = _build_fixed_sections(
            report_payload({"blue_iq": 180}, status="new_personal_best")
        )
        neutral = _build_fixed_sections(
            report_payload({"blue_iq": 180}, status="below_personal_best")
        )

        self.assertNotIn("personal-best", baseline["overall"])
        self.assertIn("new personal-best", personal_best["overall"])
        self.assertNotIn("personal-best", neutral["overall"])
        self.assertIn("This run records", neutral["overall"])


if __name__ == "__main__":
    unittest.main()
