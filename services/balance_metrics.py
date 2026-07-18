"""Independent 2D Balance measurement and scoring.

The tracker intentionally stays separate from the legacy BlueIQ Balance
formula. It measures lateral centre-of-mass control from a front-facing
camera; it does not claim to measure fore/aft or true 3D balance.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


Point = Tuple[int, int]


@dataclass(frozen=True)
class BalanceMeasurement:
    """Current Balance geometry and temporal score components."""

    com_point: Point
    support_point: Point
    plumb_point: Point
    normalized_offset: float
    offset_angle: float
    score: Optional[float]
    spread_value: Optional[float]
    spread_score: Optional[float]
    smoothness_score: Optional[float]
    rhythm_score: Optional[float]

    @property
    def range_value(self) -> Optional[float]:
        """Compatibility alias for integrations created before spread scoring."""
        return self.spread_value

    @property
    def range_score(self) -> Optional[float]:
        """Compatibility alias for integrations created before spread scoring."""
        return self.spread_score


class BalanceTracker:
    """Track and score lateral CoM movement relative to the boot midpoint."""

    # Provisional front-camera calibration constants. These are normalized by
    # visible body height and should be tuned with instructor-labelled runs.
    SPREAD_GATE_START = 0.020
    SPREAD_GATE_FULL = 0.070
    SPREAD_LOW_MARK = 0.150
    SPREAD_TRANSITION_MARK = 0.180
    SPREAD_HIGH_MARK = 0.200
    SPREAD_IDEAL = 0.250
    SPREAD_LOW_SCORE = 140.0
    SPREAD_TRANSITION_SCORE = 150.0
    SPREAD_LOW_LAMBDA = 1.5
    SCORE_MINIMUM = 60.0
    SCORE_MAXIMUM = 240.0

    def __init__(
        self,
        target_fps: float = 10.0,
        window_seconds: float = 4.0,
        trail_frames: int = 12,
    ) -> None:
        self.target_fps = max(1.0, float(target_fps))
        self.window_frames = max(12, int(round(self.target_fps * window_seconds)))
        self.minimum_samples = max(8, int(round(self.target_fps)))
        self.raw_offsets = deque(maxlen=self.window_frames)
        self.smoothed_offsets = deque(maxlen=self.window_frames)
        self.trail = deque(maxlen=max(2, int(trail_frames)))
        self.last_smoothed_offset: Optional[float] = None
        self.last_measurement: Optional[BalanceMeasurement] = None
        self.missed_frames = 0

    @staticmethod
    def _midpoint(point_a: Point, point_b: Point) -> Tuple[float, float]:
        return (
            (float(point_a[0]) + float(point_b[0])) / 2.0,
            (float(point_a[1]) + float(point_b[1])) / 2.0,
        )

    @staticmethod
    def _mean_point(points: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
        values = list(points)
        return (
            sum(point[0] for point in values) / len(values),
            sum(point[1] for point in values) / len(values),
        )

    @staticmethod
    def _as_point(point: Tuple[float, float]) -> Point:
        return int(round(point[0])), int(round(point[1]))

    @staticmethod
    def _valid_point(point: Any, width: int, height: int) -> bool:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            return False
        try:
            x_value = float(point[0])
            y_value = float(point[1])
        except (TypeError, ValueError):
            return False
        return (
            np.isfinite(x_value)
            and np.isfinite(y_value)
            and -0.05 * width <= x_value <= 1.05 * width
            and -0.05 * height <= y_value <= 1.05 * height
        )

    def _support_midpoint(
        self,
        body_points: Dict[str, Any],
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float]]:
        left_ankle = body_points.get("left_foot")
        right_ankle = body_points.get("right_foot")
        if not all(
            self._valid_point(point, width, height)
            for point in (left_ankle, right_ankle)
        ):
            return None

        left_boot_points = [left_ankle]
        right_boot_points = [right_ankle]
        left_toe = body_points.get("left_toe")
        right_toe = body_points.get("right_toe")
        if self._valid_point(left_toe, width, height):
            left_boot_points.append(left_toe)
        if self._valid_point(right_toe, width, height):
            right_boot_points.append(right_toe)

        left_boot = self._mean_point(left_boot_points)
        right_boot = self._mean_point(right_boot_points)
        return self._midpoint(left_boot, right_boot)

    @classmethod
    def _spread_quality(cls, lateral_spread: float) -> float:
        """Reward meaningful lateral movement without penalizing large turns.

        Excessive but uncontrolled movement is handled by smoothness and rhythm;
        spread itself only establishes whether meaningful movement occurred.
        """
        value = max(0.0, float(lateral_spread))

        def smoothstep(ratio: float) -> float:
            ratio = float(np.clip(ratio, 0.0, 1.0))
            return ratio * ratio * (3.0 - 2.0 * ratio)

        def score_quality(score: float) -> float:
            return (score - cls.SCORE_MINIMUM) / (
                cls.SCORE_MAXIMUM - cls.SCORE_MINIMUM
            )

        low_quality = score_quality(cls.SPREAD_LOW_SCORE)
        transition_quality = score_quality(cls.SPREAD_TRANSITION_SCORE)

        if value < cls.SPREAD_LOW_MARK:
            ratio = value / cls.SPREAD_LOW_MARK
            return low_quality * (ratio ** cls.SPREAD_LOW_LAMBDA)

        if value < cls.SPREAD_TRANSITION_MARK:
            ratio = (
                (value - cls.SPREAD_LOW_MARK)
                / (cls.SPREAD_TRANSITION_MARK - cls.SPREAD_LOW_MARK)
            )
            return low_quality + (
                transition_quality - low_quality
            ) * smoothstep(ratio)

        # Preserve the requested 20%-25% curve while holding 15%-18% in the
        # developing range. The short transition is smoothly interpolated.
        high_quality = smoothstep(cls.SPREAD_HIGH_MARK / cls.SPREAD_IDEAL)
        if value < cls.SPREAD_HIGH_MARK:
            ratio = (
                (value - cls.SPREAD_TRANSITION_MARK)
                / (cls.SPREAD_HIGH_MARK - cls.SPREAD_TRANSITION_MARK)
            )
            return transition_quality + (
                high_quality - transition_quality
            ) * smoothstep(ratio)

        return smoothstep(value / cls.SPREAD_IDEAL)

    @staticmethod
    def _smoothness_quality(values: np.ndarray, movement_range: float) -> float:
        if len(values) < 4 or movement_range <= 1e-6:
            return 0.0
        acceleration = np.diff(values, n=2)
        roughness = float(np.median(np.abs(acceleration))) / max(
            movement_range,
            0.025,
        )
        return float(np.clip(np.exp(-8.0 * roughness), 0.0, 1.0))

    def _rhythm_quality(self, values: np.ndarray) -> float:
        if len(values) < self.minimum_samples:
            return 0.50

        minimum_distance = max(3, int(round(self.target_fps * 0.30)))
        minimum_amplitude = max(0.030, 0.50 * float(np.std(values)))
        candidates = []
        for index in range(1, len(values) - 1):
            if values[index] >= values[index - 1] and values[index] > values[index + 1]:
                candidates.append((index, float(values[index]), "max"))
            elif values[index] <= values[index - 1] and values[index] < values[index + 1]:
                candidates.append((index, float(values[index]), "min"))

        extrema = []
        for candidate in candidates:
            if extrema and candidate[2] == extrema[-1][2]:
                more_extreme = (
                    candidate[1] > extrema[-1][1]
                    if candidate[2] == "max"
                    else candidate[1] < extrema[-1][1]
                )
                if more_extreme:
                    extrema[-1] = candidate
                continue
            if extrema and candidate[0] - extrema[-1][0] < minimum_distance:
                continue
            extrema.append(candidate)

        segments = []
        for left, right in zip(extrema, extrema[1:]):
            amplitude = abs(right[1] - left[1])
            duration = right[0] - left[0]
            if amplitude >= minimum_amplitude and duration >= minimum_distance:
                segments.append((amplitude, float(duration)))

        # Balance is relative to the support midpoint (zero), not relative to
        # the skier's own median. This prevents one-sided wobble from looking
        # like symmetric left/right movement.
        positive_peak = max(0.0, float(np.percentile(values, 90)))
        negative_peak = abs(min(0.0, float(np.percentile(values, 10))))
        if max(positive_peak, negative_peak) <= 1e-6:
            side_symmetry = 0.0
        else:
            side_symmetry = min(positive_peak, negative_peak) / max(
                positive_peak,
                negative_peak,
            )

        if not segments:
            # Insufficient evidence remains neutral-low rather than becoming a
            # false failure or receiving full rhythm credit.
            return 0.25 + 0.25 * side_symmetry
        if len(segments) == 1:
            return 0.35 + 0.35 * side_symmetry

        amplitudes = np.array([segment[0] for segment in segments], dtype=float)
        durations = np.array([segment[1] for segment in segments], dtype=float)
        amplitude_cv = float(np.std(amplitudes) / max(np.mean(amplitudes), 1e-6))
        duration_cv = float(np.std(durations) / max(np.mean(durations), 1e-6))
        amplitude_consistency = float(np.exp(-2.0 * amplitude_cv))
        timing_consistency = float(np.exp(-2.0 * duration_cv))

        return float(np.clip(
            0.25 * amplitude_consistency
            + 0.25 * timing_consistency
            + 0.50 * side_symmetry,
            0.0,
            1.0,
        ))

    @classmethod
    def _blueiq_score(cls, quality: float) -> float:
        return float(np.clip(
            cls.SCORE_MINIMUM
            + (cls.SCORE_MAXIMUM - cls.SCORE_MINIMUM) * quality,
            cls.SCORE_MINIMUM,
            cls.SCORE_MAXIMUM,
        ))

    def _score_components(self) -> Dict[str, Optional[float]]:
        if len(self.smoothed_offsets) < self.minimum_samples:
            return {
                "score": None,
                "spread_value": None,
                "spread_score": None,
                "smoothness_score": None,
                "rhythm_score": None,
            }

        values = np.array(self.smoothed_offsets, dtype=float)
        lateral_spread = float(np.std(values))
        robust_excursion = float(
            np.percentile(values, 90) - np.percentile(values, 10)
        )
        spread_quality = self._spread_quality(lateral_spread)
        smoothness_quality = self._smoothness_quality(values, robust_excursion)
        rhythm_quality = self._rhythm_quality(values)

        # The gate applies only to near-static movement. Once the skier has a
        # meaningful spread, the advertised 35/40/25 weights apply directly.
        gate_ratio = float(np.clip(
            (
                lateral_spread - self.SPREAD_GATE_START
            ) / (
                self.SPREAD_GATE_FULL - self.SPREAD_GATE_START
            ),
            0.0,
            1.0,
        ))
        movement_gate = gate_ratio * gate_ratio * (3.0 - 2.0 * gate_ratio)
        weighted_quality = (
            0.35 * spread_quality
            + 0.40 * smoothness_quality
            + 0.25 * rhythm_quality
        )
        overall_quality = movement_gate * weighted_quality
        return {
            "score": self._blueiq_score(overall_quality),
            "spread_value": lateral_spread,
            "spread_score": self._blueiq_score(spread_quality),
            "smoothness_score": self._blueiq_score(smoothness_quality),
            "rhythm_score": self._blueiq_score(rhythm_quality),
        }

    def update(
        self,
        body_points: Optional[Dict[str, Any]],
        frame_shape: Tuple[int, ...],
    ) -> Optional[BalanceMeasurement]:
        """Add one valid pose frame and return its Balance measurement."""
        if not body_points or len(frame_shape) < 2:
            self.miss()
            return None

        height, width = int(frame_shape[0]), int(frame_shape[1])
        required = (
            body_points.get("left_shoulder"),
            body_points.get("right_shoulder"),
            body_points.get("left_hip"),
            body_points.get("right_hip"),
        )
        if not all(self._valid_point(point, width, height) for point in required):
            self.miss()
            return None

        visibility = body_points.get("balance_visibility")
        if visibility is not None and float(visibility) < 0.50:
            self.miss()
            return None

        shoulder_mid = self._midpoint(required[0], required[1])
        hip_mid = self._midpoint(required[2], required[3])
        support_mid = self._support_midpoint(body_points, width, height)
        if support_mid is None:
            self.miss()
            return None

        # Stable 2D torso CoM proxy positioned below the torso midpoint.
        com = (
            0.30 * shoulder_mid[0] + 0.70 * hip_mid[0],
            0.30 * shoulder_mid[1] + 0.70 * hip_mid[1],
        )
        body_height = float(np.hypot(
            support_mid[0] - shoulder_mid[0],
            support_mid[1] - shoulder_mid[1],
        ))
        if body_height < max(40.0, 0.08 * height):
            self.miss()
            return None

        raw_offset = (com[0] - support_mid[0]) / body_height
        if not np.isfinite(raw_offset):
            self.miss()
            return None

        self.raw_offsets.append(float(raw_offset))
        robust_offset = float(np.median(list(self.raw_offsets)[-5:]))
        if self.last_smoothed_offset is None:
            smoothed_offset = robust_offset
        else:
            smoothed_offset = (
                0.35 * robust_offset
                + 0.65 * self.last_smoothed_offset
            )
        self.last_smoothed_offset = smoothed_offset
        self.smoothed_offsets.append(smoothed_offset)

        com_point = self._as_point(com)
        support_point = self._as_point(support_mid)
        plumb_point = (com_point[0], support_point[1])
        self.trail.append(com_point)
        self.missed_frames = 0

        vertical_distance = max(1.0, abs(support_mid[1] - com[1]))
        offset_angle = float(np.degrees(np.arctan2(
            abs(com[0] - support_mid[0]),
            vertical_distance,
        )))
        components = self._score_components()
        measurement = BalanceMeasurement(
            com_point=com_point,
            support_point=support_point,
            plumb_point=plumb_point,
            normalized_offset=smoothed_offset,
            offset_angle=offset_angle,
            score=components["score"],
            spread_value=components["spread_value"],
            spread_score=components["spread_score"],
            smoothness_score=components["smoothness_score"],
            rhythm_score=components["rhythm_score"],
        )
        self.last_measurement = measurement
        return measurement

    def miss(self) -> None:
        """Record a missing pose without adding stale data to the score."""
        self.missed_frames += 1

    def visual_measurement(self, hold_frames: int = 3) -> Optional[BalanceMeasurement]:
        """Briefly hold the last vector to avoid one-frame overlay flicker."""
        if self.last_measurement is None or self.missed_frames > hold_frames:
            return None
        return self.last_measurement

    def trail_points(self, hold_frames: int = 3) -> Tuple[Point, ...]:
        if self.visual_measurement(hold_frames) is None:
            return ()
        return tuple(self.trail)
