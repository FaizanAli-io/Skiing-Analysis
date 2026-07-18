import unittest
from types import SimpleNamespace

from services.analysis_timeline import (
    SCORING_VERSION,
    aggregate_timeline_samples,
    build_timeline_sample,
    create_timeline_payload,
    sanitize_parameter_config,
)
from services.report_generator import build_score_windows


class AnalysisTimelineTests(unittest.TestCase):
    def _sample(self, time_seconds, pressure_score, vertical_score):
        balance = SimpleNamespace(
            spread_score=180.0,
            smoothness_score=190.0,
            rhythm_score=170.0,
            spread_value=0.18,
            normalized_offset=0.04,
            offset_angle=4.2,
        )
        return build_timeline_sample(
            time_seconds=time_seconds,
            pillar_scores={
                "pressure": pressure_score,
                "balance": 175.0,
                "rotation": 185.0,
                "edging": 205.0,
            },
            score_details={
                "pressure_vertical_score": vertical_score,
                "pressure_vertical_range": 0.10,
                "pressure_knee_range_score": 160.0,
                "pressure_knee_range": 33.0,
                "rotation_separation_score": 190.0,
                "rotation_lateral_score": 170.0,
                "rotation_direction_change_score": 150.0,
                "edging_parallelism_score": 220.0,
                "edging_lateral_score": 170.0,
            },
            pressure_knee_angle=38.0,
            balance_measurement=balance,
            rotation_separation_angle=14.0,
            ski_direction_angle=28.0,
            ski_parallelism_angle=4.0,
            lateral_speed=24.0,
            upper_body_alignment=8.0,
            athletic_stance_knee=42.0,
            athletic_stance_bend=36.0,
        )

    def test_build_sample_contains_scores_values_and_report_compatibility(self):
        sample = self._sample(0.1, 170.0, 180.0)

        self.assertEqual(sample["pressure"], 170.0)
        self.assertEqual(sample["pillars"]["pressure"]["score"], 170.0)
        self.assertEqual(
            sample["pillars"]["pressure"]["components"]["vertical_range"]["value"],
            10.0,
        )
        self.assertEqual(
            sample["pillars"]["balance"]["components"]["lateral_spread"]["value"],
            18.0,
        )
        self.assertEqual(
            sample["pillars"]["rotation"]["components"]["direction_change"]["value"],
            3.0,
        )
        self.assertNotIn(
            "edge_angle",
            sample["pillars"]["edging"]["components"],
        )
        self.assertNotIn("edge_control", sample)

    def test_one_second_aggregation_averages_numeric_leaves(self):
        samples = [
            self._sample(0.1, 160.0, 170.0),
            self._sample(0.8, 180.0, 190.0),
            self._sample(1.2, 200.0, 210.0),
        ]

        result = aggregate_timeline_samples(samples, resolution="second")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["time_seconds"], 0)
        self.assertEqual(result[0]["sample_count"], 2)
        self.assertEqual(result[0]["pillars"]["pressure"]["score"], 170.0)
        self.assertEqual(
            result[0]["pillars"]["pressure"]["components"]["vertical_range"]["score"],
            180.0,
        )
        self.assertEqual(result[1]["pillars"]["pressure"]["score"], 200.0)

    def test_payload_snapshots_scoring_metadata(self):
        payload = create_timeline_payload([self._sample(0.1, 170.0, 180.0)], 10)

        self.assertEqual(payload["scoring_version"], SCORING_VERSION)
        self.assertEqual(payload["sample_rate_hz"], 10.0)
        self.assertIn("pressure", payload["parameter_config"])
        self.assertIn("value_series", payload["parameter_config"]["rotation"])
        edging_config = payload["parameter_config"]["edging"]
        self.assertNotIn(
            "edge_angle",
            {series["key"] for series in edging_config["value_series"]},
        )
        self.assertNotIn(
            "2D edge reference",
            {reference["label"] for reference in edging_config["references"]},
        )

    def test_samples_remain_compatible_with_report_windows(self):
        windows = build_score_windows([
            self._sample(1.0, 160.0, 170.0),
            self._sample(2.0, 180.0, 190.0),
        ])

        self.assertEqual(windows[0]["pillar_scores"]["pressure"], 170)
        self.assertEqual(
            windows[0]["coaching_factors"]["ski_parallel_control"],
            4,
        )

    def test_saved_edge_reference_is_removed_from_graph_config(self):
        old_config = {
            "edging": {
                "score_series": [{"key": "overall"}],
                "value_series": [
                    {"key": "parallelism"},
                    {"key": "edge_angle"},
                ],
                "references": [
                    {"label": "Ski parallelism"},
                    {"label": "2D edge reference"},
                ],
            },
        }

        sanitized = sanitize_parameter_config(old_config)

        self.assertEqual(
            [item["key"] for item in sanitized["edging"]["value_series"]],
            ["parallelism"],
        )
        self.assertEqual(
            [item["label"] for item in sanitized["edging"]["references"]],
            ["Ski parallelism"],
        )
        self.assertIn(
            "nearly the same direction",
            sanitized["edging"]["references"][0]["explanation"],
        )
        self.assertIn(
            "1 deg difference or less",
            sanitized["edging"]["references"][0]["mapping"],
        )
        self.assertEqual(len(old_config["edging"]["value_series"]), 2)

    def test_unknown_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_timeline_samples([], resolution="minute")


if __name__ == "__main__":
    unittest.main()
