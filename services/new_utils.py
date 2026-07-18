import cv2
import numpy as np
import os

from services.metric_colors import METRIC_COLORS_BGR

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

def rotate_frame(frame, angle):
    """
    Rotate frame by specified angle.
    
    Args:
        frame: Input video frame
        angle: Rotation angle (0, 90, -90, 180)
    
    Returns:
        Rotated frame
    """
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame  # Original (0°)



def getMax50Percent(arr):
    # Sort the array in descending order
    arr_sorted = sorted(arr, reverse=True)

    # Calculate the number of elements to keep (50% of total)
    num_elements = len(arr) // 2

    # Return the top 50% values
    return arr_sorted[:num_elements]

def remove_outliers(data):
    if len(data) < 4:
        return data

    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

class SkierScoring:
    def __init__(self, min_score=0, max_score=180, max_angle=20):
        self.min_score = min_score
        self.max_score = max_score
        self.max_angle = max_angle
        self.slope = (min_score - max_score) / max_angle
        self.intercept = max_score

    # def getSkiAngleScore(self, angle):
    #     if angle < 0:
    #         raise ValueError("Angle cannot be negative")

    #     # Compute score using the formula
    #     score = self.slope * angle + self.intercept

    #     # Ensure score is within range
    #     return max(self.min_score, min(int(score), self.max_score))

    def getSkiAngleScore(self,angle, max_angle=20, min_angle=1, min_score=0, max_score=180):

        if angle < 0:
          raise ValueError("Angle cannot be negative")

        if angle <= min_angle:
            return max_score
        elif angle >= max_angle:
            return min_score

        # Reverse linear interpolation
        slope = (min_score - max_score) / (max_angle - min_angle)
        score = slope * (angle - min_angle) + max_score

        return int(score)


    def getLateralMovementScore(
        self,
        speed,
        max_score=180,
        min_score=0,
        min_speed=0,
        max_speed=45,
        low_speed_threshold=30,
        low_speed_lambda=2.0,
    ):
        """Score lateral speed with a stronger penalty below 30 px/frame.

        The returned 0-180 score is later mapped to the 60-240 Blue IQ scale.
        Thirty px/frame maps to the midpoint (150 Blue IQ), while 45 or more
        maps to 240.
        """
        if speed < 0:
            return min_score

        if speed <= min_speed:
            return min_score
        elif speed >= max_speed:
            return max_score

        threshold = min(max(float(low_speed_threshold), min_speed), max_speed)
        midpoint_score = min_score + 0.5 * (max_score - min_score)
        if speed <= threshold:
            ratio = (speed - min_speed) / max(threshold - min_speed, 1e-6)
            score = min_score + (
                midpoint_score - min_score
            ) * (ratio ** low_speed_lambda)
        else:
            ratio = (speed - threshold) / max(max_speed - threshold, 1e-6)
            score = midpoint_score + ratio * (max_score - midpoint_score)

        return int(score)

    def getShoulderHipScore(self,angle, max_score=180, min_score=0, min_angle=8, max_angle=20):
        if angle < 0:
            raise ValueError("Angle cannot be negative")

        if angle <= min_angle:
            return min_score
        elif angle >= max_angle:
            return max_score

        # Linear interpolation
        slope = (max_score - min_score) / (max_angle - min_angle)
        score = slope * (angle - min_angle) + min_score

        return int(score)

    def getRotationSeparationScore(self, angle, max_angle=8, min_score=0, max_score=180):
        """Score separation with a steeper penalty below four degrees.

        Internal scores map to Blue IQ by adding 60:
        0 degrees -> 60, 4 degrees -> 180, 8+ degrees -> 240.
        """
        if angle is None or angle <= 0:
            return min_score
        if angle >= max_angle:
            return max_score

        breakpoint_angle = 4
        breakpoint_score = 120  # Maps to 180 on the 60-240 Blue IQ scale.
        if angle <= breakpoint_angle:
            score = (angle / breakpoint_angle) * breakpoint_score
        else:
            upper_ratio = (angle - breakpoint_angle) / (max_angle - breakpoint_angle)
            score = breakpoint_score + upper_ratio * (max_score - breakpoint_score)
        return int(score)

    def getRotationDynamicsScore(self, angles, target_fps=10):
        """Score cyclic 2D lower-body lead dynamics on the 60-240 scale."""
        values = [float(value) for value in angles if value is not None and np.isfinite(value)]
        if len(values) < 3:
            return 60

        # Five-sample Hampel filtering removes isolated pose-estimation jumps
        # without flattening legitimate turn peaks and troughs.
        smoothed = []
        for index in range(len(values)):
            start = max(0, index - 2)
            end = min(len(values), index + 3)
            window = np.array(values[start:end], dtype=float)
            median = float(np.median(window))
            mad = float(np.median(np.abs(window - median)))
            threshold = max(3.0, 3.0 * 1.4826 * mad)
            value = values[index]
            smoothed.append(median if abs(value - median) > threshold else value)

        extrema = []
        first_type = "max" if smoothed[0] >= smoothed[1] else "min"
        extrema.append((0, smoothed[0], first_type))
        for index in range(1, len(smoothed) - 1):
            previous_value = smoothed[index - 1]
            value = smoothed[index]
            next_value = smoothed[index + 1]
            if value >= previous_value and value > next_value:
                extrema.append((index, value, "max"))
            elif value <= previous_value and value < next_value:
                extrema.append((index, value, "min"))
        last_type = "max" if smoothed[-1] >= smoothed[-2] else "min"
        extrema.append((len(smoothed) - 1, smoothed[-1], last_type))

        # Collapse adjacent extrema of the same type to the more extreme value.
        alternating = []
        for extremum in extrema:
            if alternating and alternating[-1][2] == extremum[2]:
                replace = (
                    extremum[1] > alternating[-1][1]
                    if extremum[2] == "max"
                    else extremum[1] < alternating[-1][1]
                )
                if replace:
                    alternating[-1] = extremum
            else:
                alternating.append(extremum)

        minimum_distance = max(2, int(round(target_fps * 0.2)))
        valid_segments = []
        for left, right in zip(alternating, alternating[1:]):
            frame_distance = right[0] - left[0]
            amplitude = abs(right[1] - left[1])
            if frame_distance >= minimum_distance and amplitude >= 2.0:
                valid_segments.append((left[0], right[0], amplitude))

        if not valid_segments:
            return 60

        amplitudes = np.array([segment[2] for segment in valid_segments], dtype=float)
        representative_amplitude = float(np.median(amplitudes))
        amplitude_quality = float(np.clip((representative_amplitude - 5.0) / 11.0, 0.0, 1.0))

        correct_steps = 0
        total_steps = 0
        for start, end, _ in valid_segments:
            expected_direction = 1 if smoothed[end] > smoothed[start] else -1
            for index in range(start + 1, end + 1):
                delta = smoothed[index] - smoothed[index - 1]
                if abs(delta) <= 0.5 or delta * expected_direction > 0:
                    correct_steps += 1
                total_steps += 1
        smoothness = correct_steps / total_steps if total_steps else 0.0

        amplitude_mean = float(np.mean(amplitudes))
        amplitude_cv = float(np.std(amplitudes) / amplitude_mean) if amplitude_mean > 0 else 1.0
        consistency = float(np.exp(-2.0 * amplitude_cv))

        durations = np.array([segment[1] - segment[0] for segment in valid_segments], dtype=float)
        duration_mean = float(np.mean(durations))
        duration_cv = float(np.std(durations) / duration_mean) if duration_mean > 0 else 1.0
        rhythm = float(np.exp(-2.0 * duration_cv))

        dynamics_quality = amplitude_quality * (
            0.60
            + 0.20 * smoothness
            + 0.12 * consistency
            + 0.08 * rhythm
        )
        return float(np.clip(60.0 + 180.0 * dynamics_quality, 60.0, 240.0))

    def getPressureAngleScore(self,angle, max_angle=28, min_angle=5, min_score=0, max_score=40):

        if angle < 0:
            raise ValueError("Angle cannot be negative")

        if angle <= min_angle:
            return min_score
        elif angle >= max_angle:
            return max_score

        # Linear interpolation
        slope = (max_score - min_score) / (max_angle - min_angle)
        score = slope * (angle - min_angle) + min_score

        return int(score)

    def getPressureVerticalRange(self, relative_heights):
        """Return robust P90-P10 pelvis travel as a fraction of body height."""
        values = [
            float(value) for value in relative_heights
            if value is not None and np.isfinite(value)
        ]
        if len(values) < 3:
            return 0.0

        filtered = []
        for index, value in enumerate(values):
            start = max(0, index - 2)
            end = min(len(values), index + 3)
            window = np.array(values[start:end], dtype=float)
            median = float(np.median(window))
            mad = float(np.median(np.abs(window - median)))
            threshold = max(0.015, 3.0 * 1.4826 * mad)
            filtered.append(median if abs(value - median) > threshold else value)

        return max(
            0.0,
            float(np.percentile(filtered, 90) - np.percentile(filtered, 10)),
        )

    def getPressureVerticalRangeScore(
        self,
        relative_heights,
        minimum_useful_range=0.05,
        target_range=0.12,
    ):
        """Map normalized vertical travel linearly onto the 60-240 scale."""
        vertical_range = self.getPressureVerticalRange(relative_heights)
        quality = np.clip(
            (vertical_range - minimum_useful_range)
            / (target_range - minimum_useful_range),
            0.0,
            1.0,
        )
        return float(60.0 + 180.0 * quality)

    def getPressureKneeAngleRange(self, knee_angles):
        """Return robust P90-P10 flexion/extension range in degrees."""
        values = [
            float(value) for value in knee_angles
            if value is not None and np.isfinite(value)
        ]
        if len(values) < 3:
            return 0.0

        filtered = []
        for index, value in enumerate(values):
            start = max(0, index - 2)
            end = min(len(values), index + 3)
            window = np.array(values[start:end], dtype=float)
            median = float(np.median(window))
            mad = float(np.median(np.abs(window - median)))
            threshold = max(5.0, 3.0 * 1.4826 * mad)
            filtered.append(median if abs(value - median) > threshold else value)

        return max(
            0.0,
            float(np.percentile(filtered, 90) - np.percentile(filtered, 10)),
        )

    def getPressureKneeRangeScore(
        self,
        knee_angles,
        minimum_useful_range=18.0,
        target_range=45.0,
    ):
        """Map knee flexion/extension range linearly onto the 60-240 scale."""
        knee_range = self.getPressureKneeAngleRange(knee_angles)
        quality = np.clip(
            (knee_range - minimum_useful_range)
            / (target_range - minimum_useful_range),
            0.0,
            1.0,
        )
        return float(60.0 + 180.0 * quality)

    def getTiltScore(self,tilt_ratio, min_score=0, max_score=40):

        if not 0 <= tilt_ratio <= 1:
            return 0

        score = tilt_ratio * (max_score - min_score) + min_score
        return int(score)

    def calculateTiltRatio(self,total_turns,tilt_count):
        if total_turns == 0:
            return 0
        return tilt_count/total_turns



    def getSkiAngle2Score(self, angles):
        if len(angles) < 2:
            return 0  # Not enough data to calculate change

        # Take the last 10 values (or less if array is smaller)
        last_angles = angles[-10:]

        total_change = 0
        miss = 0

        for i in range(1, len(last_angles)):
            angle2, time2 = last_angles[i]
            angle1, time1 = last_angles[i-1]
            change = abs((angle2-angle1)/(time2-time1))
            if change > 15:
                miss+=1
            else:
                total_change += change

        avg_change = total_change/(len(last_angles) - 1 - miss)

        score = avg_change * 30

        score = min(score, 180)

        score = max(0, score)
        score = min(score, 180)
        return score

    def getBodyAngleScore(self, angle):
        x = 180 / (45 - 85)
        score = 180 + x*(angle - 45)
        # score = 4 * (90 - angle)
        score = max(0, score)
        score = min(score, 180)
        return score

    def getKneeAngleScore(self, angle, skiAngle):
        if skiAngle > 20:
            return 0
        score = 2.5 * (angle)
        score = max(0, score)
        score = min(score, 180)
        return score

    # def getLateralMovementScore(self, lateral_movement, score_lateralMovement):
    #     print("LATERAL MOVEMENT")
    #     print(lateral_movement)
    #     #score = lateral_movement * 10
    #     x = 180 / (40-15)
    #     score = x*(lateral_movement - 15)
    #     score = min(score, 180)
    #     #score = score_lateralMovement + 0.2*(score-score_lateralMovement)
    #     score = max(0, score)
    #     score = min(score, 180)
    #     return score

    def mapScore(self, score):
        if score is None:
            return 60
        return min(240, score + 60)

    def getAvg(self, arr):
        count = len(arr)
        sum = 0
        for i in arr:
            sum=round(sum + i, 1)
        avg = round(sum/count, 1)
        # avg = int(avg)
        return avg

    def getSide(self,person_x, reference_x, curr_side):
        if person_x < reference_x:
            return "Left"
        elif person_x > reference_x:
            return "Right"
        else:
            return curr_side

    def getTurnDurations(self, angles, startingAngle=35, stoppingAngle=40):
        status = "Moving"
        times = []
        start = None
        stop = None
        turn = 0

        for angle, time in angles:
            time /=10
            if status == "Moving" and angle < startingAngle:
                status = "Turning"
                start = time
            elif status == "Turning" and angle > stoppingAngle and time - start > 0.2:
                status = "Moving"
                timeTaken = round(time - start, 1)
                start = None
                times.append(timeTaken)
                turn += 1
        return times

    def getNoOfTurnsScore(self, timePerTurn):
        # score = 230 - timePerTurn * 50
        x = 180 / (1.5 - 4)
        score = 180 + x*(timePerTurn - 1.5)
        score = max(0, score)
        score = min(score, 180)
        return score

    def getTurnTimeScore(self, avgTurningTime):
        x = 180 / (0.25 - 0.6)
        score = 180 + x*(avgTurningTime - 0.25)
        # score = avgTurningTime * 250
        # print(score)
        # score = 210 - score
        score = max(0, score)
        score = min(score, 180)
        return score

    def getFramesInTurn(self,turns,fps):
      total = 0
      tot_fps = fps + (fps/5)
      for i in range(turns):
        total += tot_fps
      return total





def calculate_angle(line1, line2):
    """Calculate the angle between two lines given their endpoints."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute direction vectors
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Compute dot product and magnitudes
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    # Compute the angle in radians and convert to degrees
    cos_theta = dot_product / (mag_v1 * mag_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg = abs(180 - angle_deg)

    return angle_deg


def calculate_signed_vertical_angle(start, end, vertical_direction="down"):
    """Return a signed 2D angle relative to the natural vertical direction."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if vertical_direction == "up":
        return float(np.degrees(np.arctan2(dx, -dy)))
    return float(np.degrees(np.arctan2(dx, dy)))



def resize_and_center_frame(frame, target_width, target_height):

    original_height, original_width = frame.shape[:2]

    # Compute scaling factor (avoid zoom-in)
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)

    # Resize the frame with aspect ratio maintained
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create a blank frame of target size and center the resized frame
    normalized_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    normalized_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    return normalized_frame, new_width, new_height

def detect_people_and_skis_from_results(results, scores, history):
    """
    Processes detection results to extract people and ski equipment.

    Args:
        results: YOLO model tracking results.
        scores (dict): Score tracking for each person (track_id).
        history (dict): Movement history tracking for each person (track_id).

    Returns:
        tuple: (detected_people, ski_boxes, detected_ids)
    """
    detected_people = []
    ski_boxes = []

    if results[0].boxes.id is not None:
        for box, class_id, conf, track_id in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
            results[0].boxes.id.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, box)

            if int(class_id) == 0 and conf > 0.5:  # Person
                width, height = x2 - x1, y2 - y1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                detected_people.append((track_id, x1, y1, x2, y2, width, height, center_x, center_y))

                if track_id not in scores:
                    scores[track_id] = 100
                    history[track_id] = []

                history[track_id].append((center_x, center_y, width, height))

            elif int(class_id) in [30, 31, 36] and conf > 0.1:  # Ski objects
                ski_boxes.append((x1, y1, x2, y2))

    detected_ids = {p[0] for p in detected_people}
    return detected_people, ski_boxes, detected_ids


def update_turns_and_side(track_id, history, image_centre, side, turns, scoring):
    if len(history[track_id]) > 1:
        prev_x, prev_y, _, _ = history[track_id][-2]
        curr_x, curr_y, _, _ = history[track_id][-1]

        prev_side = side
        side = scoring.getSide(curr_x, image_centre, side)

        if side is not None and prev_side is not None and side != prev_side:
            turns += 1

    return side, turns


def draw_pose_connections(frame, coords):
    (
        navel_x,navel_y, left_foot_x,left_foot_y, right_foot_x,right_foot_y,  left_knee_x, left_knee_y,
        right_knee_x, right_knee_y,left_shoulder_x, left_shoulder_y,  right_shoulder_x,
        right_shoulder_y, left_hip_x, left_hip_y, right_hip_x, right_hip_y
    ) = coords


    # Keep pose context deliberately faint so metric colors remain dominant.
    pose_layer = frame.copy()
    pose_color = (190, 190, 190)
    keypoints = (
        (navel_x, navel_y),
        (left_foot_x, left_foot_y),
        (right_foot_x, right_foot_y),
        (left_knee_x, left_knee_y),
        (right_knee_x, right_knee_y),
        (left_hip_x, left_hip_y),
        (right_hip_x, right_hip_y),
    )
    connections = (
        ((navel_x, navel_y), (left_knee_x, left_knee_y)),
        ((left_knee_x, left_knee_y), (left_foot_x, left_foot_y)),
        ((navel_x, navel_y), (right_knee_x, right_knee_y)),
        ((right_knee_x, right_knee_y), (right_foot_x, right_foot_y)),
        ((left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y)),
        ((left_hip_x, left_hip_y), (right_hip_x, right_hip_y)),
        ((left_shoulder_x, left_shoulder_y), (left_hip_x, left_hip_y)),
        ((right_shoulder_x, right_shoulder_y), (right_hip_x, right_hip_y)),
        ((left_hip_x, left_hip_y), (left_knee_x, left_knee_y)),
        ((right_hip_x, right_hip_y), (right_knee_x, right_knee_y)),
    )
    for point in keypoints:
        cv2.circle(pose_layer, point, 3, pose_color, -1, cv2.LINE_AA)
    for start, end in connections:
        cv2.line(pose_layer, start, end, pose_color, 1, cv2.LINE_AA)
    cv2.addWeighted(pose_layer, 0.28, frame, 0.72, 0, frame)

    return frame


def _draw_biomech_label(frame, text, x, y, color=(180, 230, 255)):
    """Draw a compact readable label for coach-mode biomechanics overlays."""
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(4, min(x, frame.shape[1] - tw - 12))
        y = max(th + 8, min(y, frame.shape[0] - 8))
        cv2.rectangle(frame, (x - 5, y - th - 6), (x + tw + 7, y + baseline + 5), (8, 18, 34), -1)
        cv2.rectangle(frame, (x - 5, y - th - 6), (x + tw + 7, y + baseline + 5), color, 1)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    except Exception:
        pass


def _line_midpoint(line):
    x1, y1, x2, y2 = line
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _draw_joint_angle_arc(frame, vertex, point_a, point_b, color, radius=24):
    """Draw the shorter angle arc between two segments at a joint."""
    angle_a = np.degrees(np.arctan2(point_a[1] - vertex[1], point_a[0] - vertex[0]))
    angle_b = np.degrees(np.arctan2(point_b[1] - vertex[1], point_b[0] - vertex[0]))
    sweep = (angle_b - angle_a) % 360
    if sweep > 180:
        angle_a, angle_b = angle_b, angle_a
        sweep = 360 - sweep
    cv2.ellipse(
        frame,
        vertex,
        (radius, radius),
        0,
        angle_a,
        angle_a + sweep,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_metric_legend(frame):
    """Draw the four metric identities using the shared vector colors."""
    items = (
        ("EDGING", METRIC_COLORS_BGR["edging"]),
        ("PRESSURE", METRIC_COLORS_BGR["pressure"]),
        ("ROTATION", METRIC_COLORS_BGR["rotation"]),
        ("BALANCE", METRIC_COLORS_BGR["balance"]),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.34
    thickness = 1
    gap = 14
    dot_diameter = 7
    text_widths = [
        cv2.getTextSize(label, font, scale, thickness)[0][0]
        for label, _ in items
    ]
    content_width = sum(dot_diameter + 5 + width for width in text_widths)
    content_width += gap * (len(items) - 1)
    box_width = content_width + 20
    box_height = 28
    x = max(8, frame.shape[1] - box_width - 12)
    y = 12

    panel = frame.copy()
    cv2.rectangle(panel, (x, y), (x + box_width, y + box_height), (8, 18, 34), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    cursor_x = x + 10
    baseline_y = y + 19
    for (label, color), text_width in zip(items, text_widths):
        cv2.circle(
            frame,
            (cursor_x + dot_diameter // 2, y + box_height // 2),
            dot_diameter // 2,
            color,
            -1,
            cv2.LINE_AA,
        )
        cursor_x += dot_diameter + 5
        cv2.putText(
            frame,
            label,
            (cursor_x, baseline_y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_x += text_width + gap


def draw_biomechanics_overlay(frame, body_points, ski_lines, metrics):
    """Draw coach-mode visual guides for the live biomechanics values.

    This is display-only and does not alter scoring inputs.
    """
    try:
        body_points = body_points or {}
        guide = (190, 190, 190)
        edging_color = METRIC_COLORS_BGR["edging"]
        pressure_color = METRIC_COLORS_BGR["pressure"]
        rotation_color = METRIC_COLORS_BGR["rotation"]
        balance_color = METRIC_COLORS_BGR["balance"]
        white = (245, 250, 255)

        # Body alignment: shoulder midpoint to hip midpoint.
        left_shoulder = body_points.get("left_shoulder")
        right_shoulder = body_points.get("right_shoulder")
        left_hip = body_points.get("left_hip")
        right_hip = body_points.get("right_hip")
        left_knee = body_points.get("left_knee")
        right_knee = body_points.get("right_knee")
        left_foot = body_points.get("left_foot")
        right_foot = body_points.get("right_foot")
        navel = body_points.get("navel")

        # Rotation proxy: midpoint upper/lower-body lead relative to vertical.
        if all((left_shoulder, right_shoulder, left_hip, right_hip, left_foot, right_foot)):
            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_mid = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
            boot_mid = ((left_foot[0] + right_foot[0]) // 2, (left_foot[1] + right_foot[1]) // 2)

            cv2.line(
                frame,
                (hip_mid[0], shoulder_mid[1] - 12),
                (hip_mid[0], boot_mid[1] + 12),
                guide,
                1,
                cv2.LINE_AA,
            )
            cv2.line(frame, hip_mid, shoulder_mid, rotation_color, 3, cv2.LINE_AA)
            cv2.line(frame, hip_mid, boot_mid, rotation_color, 3, cv2.LINE_AA)
            cv2.circle(frame, shoulder_mid, 5, rotation_color, -1, cv2.LINE_AA)
            cv2.circle(frame, hip_mid, 5, rotation_color, -1, cv2.LINE_AA)
            cv2.circle(frame, boot_mid, 5, rotation_color, -1, cv2.LINE_AA)

            rotation_angle = metrics.get("rotation_angle")
            if isinstance(rotation_angle, (int, float)):
                _draw_biomech_label(
                    frame,
                    f"Rotation {rotation_angle:.0f} deg",
                    hip_mid[0] + 16,
                    hip_mid[1] - 14,
                    rotation_color,
                )

        # Balance: CoM proxy, vertical plumb, support offset, and short trail.
        balance_com = metrics.get("balance_com")
        balance_support = metrics.get("balance_support")
        balance_plumb = metrics.get("balance_plumb")
        balance_trail = metrics.get("balance_trail") or ()
        if balance_com and balance_support and balance_plumb:
            trail_layer = frame.copy()
            trail_count = len(balance_trail)
            for index, trail_point in enumerate(balance_trail):
                strength = (index + 1) / max(1, trail_count)
                trail_color = tuple(
                    int(channel * (0.30 + 0.70 * strength))
                    for channel in balance_color
                )
                cv2.circle(
                    trail_layer,
                    trail_point,
                    max(2, int(round(2 + 2 * strength))),
                    trail_color,
                    -1,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(trail_layer, 0.60, frame, 0.40, 0, frame)

            cv2.line(
                frame,
                balance_com,
                balance_plumb,
                balance_color,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                balance_plumb,
                balance_support,
                balance_color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(frame, balance_com, 7, white, -1, cv2.LINE_AA)
            cv2.circle(frame, balance_com, 7, balance_color, 2, cv2.LINE_AA)
            cv2.circle(frame, balance_support, 5, balance_color, -1, cv2.LINE_AA)
            cv2.circle(frame, balance_plumb, 4, balance_color, -1, cv2.LINE_AA)

            balance_angle = metrics.get("balance_angle")
            if isinstance(balance_angle, (int, float)):
                _draw_biomech_label(
                    frame,
                    f"Balance {balance_angle:.0f} deg",
                    balance_com[0] + 14,
                    balance_com[1] - 16,
                    balance_color,
                )

        # Pressure: thigh and shin vectors with knee flexion angle.
        if left_hip and left_knee and left_foot:
            cv2.line(frame, left_hip, left_knee, pressure_color, 3, cv2.LINE_AA)
            cv2.line(frame, left_knee, left_foot, pressure_color, 3, cv2.LINE_AA)
            cv2.circle(frame, left_knee, 5, pressure_color, -1, cv2.LINE_AA)
            _draw_joint_angle_arc(frame, left_knee, left_hip, left_foot, pressure_color)
        if right_hip and right_knee and right_foot:
            cv2.line(frame, right_hip, right_knee, pressure_color, 3, cv2.LINE_AA)
            cv2.line(frame, right_knee, right_foot, pressure_color, 3, cv2.LINE_AA)
            cv2.circle(frame, right_knee, 5, pressure_color, -1, cv2.LINE_AA)
            _draw_joint_angle_arc(frame, right_knee, right_hip, right_foot, pressure_color)

            pressure_angle = metrics.get("pressure_angle")
            if isinstance(pressure_angle, (int, float)):
                _draw_biomech_label(
                    frame,
                    f"Pressure {pressure_angle:.0f} deg",
                    right_knee[0] + 14,
                    right_knee[1] + 10,
                    pressure_color,
                )

        # Recent vertical hip travel, displayed as a compact range bar.
        pressure_range = metrics.get("pressure_vertical_range")
        if right_hip and pressure_range and len(pressure_range) == 3:
            range_min, range_max, current = pressure_range
            bar_x = min(frame.shape[1] - 12, right_hip[0] + 42)
            bar_top = max(12, right_hip[1] - 32)
            bar_bottom = min(frame.shape[0] - 12, right_hip[1] + 32)
            cv2.line(frame, (bar_x, bar_top), (bar_x, bar_bottom), pressure_color, 2, cv2.LINE_AA)
            cv2.line(frame, (bar_x - 5, bar_top), (bar_x + 5, bar_top), pressure_color, 2, cv2.LINE_AA)
            cv2.line(frame, (bar_x - 5, bar_bottom), (bar_x + 5, bar_bottom), pressure_color, 2, cv2.LINE_AA)
            if range_max > range_min:
                ratio = (current - range_min) / (range_max - range_min)
            else:
                ratio = 0.5
            # Larger relative height means the pelvis is higher, so invert the
            # screen-space Y direction when positioning the marker.
            marker_y = int(bar_top + (1.0 - ratio) * (bar_bottom - bar_top))
            cv2.circle(frame, (bar_x, marker_y), 5, pressure_color, -1, cv2.LINE_AA)

        # Edging proxy: ski parallelism, measured as the angle between skis.
        for line in ski_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), edging_color, 3, cv2.LINE_AA)
        if len(ski_lines) >= 2:
            mid_a = _line_midpoint(ski_lines[0])
            mid_b = _line_midpoint(ski_lines[1])
            cv2.line(frame, mid_a, mid_b, edging_color, 2, cv2.LINE_AA)
            cv2.circle(frame, mid_a, 4, edging_color, -1, cv2.LINE_AA)
            cv2.circle(frame, mid_b, 4, edging_color, -1, cv2.LINE_AA)
            ski_parallelism = metrics.get("ski_separation")
            if isinstance(ski_parallelism, (int, float)):
                label_x = (mid_a[0] + mid_b[0]) // 2 + 12
                label_y = min(mid_a[1], mid_b[1]) - 14
                _draw_biomech_label(
                    frame,
                    f"Edging {ski_parallelism:.0f} deg",
                    label_x,
                    label_y,
                    edging_color,
                )

        _draw_metric_legend(frame)

    except Exception:
        return frame

    return frame


def calculate_adjusted_speed(left_hip, right_hip, prev_point, speed_list, actual_speed):
    # Compute midpoint (current point)
    current_point = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
    speed = None

    if prev_point is not None:
        # Simple x-axis speed calculation
        speed = abs(current_point[0] - prev_point[0])
        actual_speed.append(speed)

        # Analyze last 20 speeds (or fewer)
        last20_speeds = speed_list[-20:] if len(speed_list) >= 5 else speed_list
        recent_speeds = sorted(last20_speeds, reverse=True)[:10]

        if recent_speeds:
            avg_speed = sum(recent_speeds) / len(recent_speeds)
            if speed < avg_speed:
                adjusted_speed = (speed + avg_speed) / 2
                speed = adjusted_speed
                speed_list.append(adjusted_speed)
            else:
                speed_list.append(speed)
        else:
            speed_list.append(speed)  # For the first few frames

    return current_point, speed, speed_list, actual_speed


def check_tilting(frame, speed, hipVertAngle, speed_threshold=18.0, angle_threshold=30.0):
    """
    Checks if the subject is tilting based on speed and hip angle.
    Draws appropriate text on the frame and returns a tilting flag.

    Returns:
        1 if tilting, 0 otherwise
    """
    if speed is not None and speed > speed_threshold and hipVertAngle > angle_threshold:
        # cv2.putText(frame, "Tilting", (125, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return 1
    else:
        # cv2.putText(frame, "Not Tilting", (125, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return 0


def detect_ski_lines(frame, ski_boxes, flag, rotation_angle=0):
    """
    Detects and draws ski lines from bounding boxes on the frame.

    Args:
        frame (np.ndarray): The image frame.
        ski_boxes (list of tuples): List of bounding boxes [(x1, y1, x2, y2), ...]
        flag (bool): Flag indicating if skis were detected
        rotation_angle (int): The rotation angle applied to the video (0, 90, -90, 180)

    Returns:
        tuple: (list of ski lines, flag)
    """
    ski_lines = []

    if ski_boxes:
        # Get the overall bounding box
        x_min = min([box[0] for box in ski_boxes])
        y_min = min([box[1] for box in ski_boxes])
        x_max = max([box[2] for box in ski_boxes])
        y_max = max([box[3] for box in ski_boxes])

        # Split the bounding box into two halves (left and right ski)
        # Skis are always side by side (left foot, right foot) regardless of rotation
        # because rotation is already applied to the frame before ski detection
        mid_x = (x_min + x_max) // 2
        ski_boxes = [(x_min, y_min, mid_x, y_max), (mid_x, y_min, x_max, y_max)]

        if ski_boxes:
            flag = True

        for idx, (x1, y1, x2, y2) in enumerate(ski_boxes):
            # Keep detection boxes hidden for client-facing output; ski vector lines below remain visible.
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ski_roi = frame[y1:y2, x1:x2]

            if ski_roi.size == 0:
                continue

            # Preprocessing
            gray_ski = cv2.cvtColor(ski_roi, cv2.COLOR_BGR2GRAY)
            gray_ski = cv2.GaussianBlur(gray_ski, (5, 5), 0)

            # Mask to avoid edges near borders
            mask = np.zeros_like(gray_ski)
            cv2.rectangle(mask, (5, 5), (gray_ski.shape[1] - 5, gray_ski.shape[0] - 5), 255, -1)

            # Canny edge detection
            edges = cv2.Canny(gray_ski, 50, 150)
            edges = cv2.bitwise_and(edges, edges, mask=mask)

            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=50, maxLineGap=20)

            if lines is not None:
                # Pick the longest line
                longest_line = max(lines, key=lambda line: np.linalg.norm(np.array(line[0][:2]) - np.array(line[0][2:])))
                x1_l, y1_l, x2_l, y2_l = longest_line[0]

                # Shift coordinates back to original frame
                x1_l, x2_l = x1_l + x1, x2_l + x1
                y1_l, y2_l = y1_l + y1, y2_l + y1

                cv2.line(
                    frame,
                    (x1_l, y1_l),
                    (x2_l, y2_l),
                    METRIC_COLORS_BGR["edging"],
                    4,
                    cv2.LINE_AA,
                )

                ski_lines.append((x1_l, y1_l, x2_l, y2_l))

    return ski_lines,flag


def calculate_blue_scores(
    turns,
    duration,
    TskiAngles2,
    edging_angle_score,
    lateral_score,
    bending_angle_score,
    knee_angle_score,
    skiAngle2_score,
    pressure_angle_score,
    pressure_speed_score,
    pressure_relative_heights,
    pressure_knee_angles,
    balance_score,
    rotation_separation_score,
    hip_shoulder,
    scoring,
    target_fps,
    count_tilt,
    score_details=None,
):
    # Turn-based scores
    if turns == 0:
        score_noOfTurns = 0
        score_turnTime = 0
    else:
        timePerTurn = duration / turns
        # print(f"{timePerTurn} seconds per turn")
        # print(f"{TskiAngles2} ANALYZE FOR TURN DURATION")

        turningTimes = scoring.getTurnDurations(TskiAngles2)
        turningTimes = remove_outliers(turningTimes)

        if turningTimes:
            avgTurningTime = round(sum(turningTimes) / len(turningTimes), 2)
            # print(f"Average Turning Time: {avgTurningTime}")

            score_turnTime = scoring.getTurnTimeScore(avgTurningTime)
            score_turnTime = scoring.mapScore(score_turnTime)
            # print(f"score_turnTime: {score_turnTime}")

            score_noOfTurns = scoring.getNoOfTurnsScore(timePerTurn)
            score_noOfTurns = scoring.mapScore(score_noOfTurns)
            # print(f"score_noOfTurns: {score_noOfTurns}")
        else:
            score_turnTime = 0
            score_noOfTurns = 0

    # Smooth the current lateral signal over at most five valid measurements.
    # During startup, use every available value; before the first valid speed,
    # use the 60-point Blue IQ baseline.
    recent_lateral_scores = lateral_score[-5:]
    recent_lateral_score = (
        sum(recent_lateral_scores) / len(recent_lateral_scores)
        if recent_lateral_scores else 60.0
    )

    # Removing outliers
    edging_angle_score = remove_outliers(edging_angle_score)
    bending_angle_score = remove_outliers(bending_angle_score)
    knee_angle_score = remove_outliers(knee_angle_score)
    skiAngle2_score = remove_outliers(skiAngle2_score)
    pressure_angle_score = remove_outliers(pressure_angle_score)
    pressure_speed_score = remove_outliers(pressure_speed_score)
    rotation_separation_score = remove_outliers(rotation_separation_score)

    knee_angle_score = getMax50Percent(knee_angle_score)
    bending_angle_score = getMax50Percent(bending_angle_score)

    # Averages
    avg_edging_angle_score = sum(edging_angle_score) / len(edging_angle_score) if edging_angle_score else 0
    avg_bending_angle_score = sum(bending_angle_score) / len(bending_angle_score) if bending_angle_score else 0
    avg_knee_angle_score = sum(knee_angle_score) / len(knee_angle_score) if knee_angle_score else 0
    avg_skiAngle2_score = sum(skiAngle2_score) / len(skiAngle2_score) if skiAngle2_score else 0
    avg_pressure_angle_score = sum(pressure_angle_score) / len(pressure_angle_score) if pressure_angle_score else 0
    avg_pressure_speed_score = sum(pressure_speed_score) / len(pressure_speed_score) if pressure_speed_score else 0
    avg_rotation_separation_score = (
        sum(rotation_separation_score) / len(rotation_separation_score)
        if rotation_separation_score else None
    )
    avg_shoulder_hip_score = sum(hip_shoulder) / len(hip_shoulder) if hip_shoulder else 0

    # print(f"Avg hip-shoulder angle score : {avg_shoulder_hip_score}")

    # Edging scoring
    Blue_edging_score = (
        0.70 * avg_edging_angle_score +
        0.30 * recent_lateral_score
    )
    Blue_edging_final = Blue_edging_score

    # Legacy Balance scoring retained for comparison and rollback.
    # Blue_balance_score = (
    #     0.3 * avg_bending_angle_score
    #     + 0.3 * avg_knee_angle_score
    #     + 0.3 * score_turnTime
    #     + 0.1 * avg_shoulder_hip_score
    # )
    # Blue_balance_final = Blue_balance_score

    # The active Balance score is calculated independently by BalanceTracker
    # from normalized CoM lateral spread, smoothness, and rhythm.
    Blue_balance_final = (
        float(np.clip(balance_score, 60.0, 240.0))
        if isinstance(balance_score, (int, float)) and np.isfinite(balance_score)
        else None
    )

    # Rotation scoring: body-separation dynamics is the primary signal.
    rotation_separation_component = (
        avg_rotation_separation_score
        if avg_rotation_separation_score is not None
        else 60.0
    )
    Blue_rotation_final = float(np.clip(
        0.60 * rotation_separation_component
        + 0.30 * recent_lateral_score
        + 0.10 * avg_skiAngle2_score,
        60.0,
        240.0,
    ))

    # Legacy Pressure scoring retained for comparison and rollback.
    # total_frames = scoring.getFramesInTurn(turns, target_fps)
    # ratio = scoring.calculateTiltRatio(total_frames, count_tilt)
    # tilt_score = scoring.getTiltScore(ratio)
    # Blue_pressure_score = (
    #     avg_pressure_angle_score + avg_pressure_speed_score + tilt_score
    # )
    # Blue_pressure_final = scoring.mapScore(Blue_pressure_score)

    # New Pressure proxies. A static crouch has near-zero movement range and
    # therefore cannot receive a high score from knee flexion alone.
    pressure_vertical_score = scoring.getPressureVerticalRangeScore(
        pressure_relative_heights
    )
    pressure_knee_range_score = scoring.getPressureKneeRangeScore(
        pressure_knee_angles
    )
    vertical_quality = (pressure_vertical_score - 60.0) / 180.0
    knee_range_quality = (pressure_knee_range_score - 60.0) / 180.0

    pressure_quality = (
        0.75 * vertical_quality
        + 0.25 * knee_range_quality
    )
    Blue_pressure_final = float(np.clip(
        60.0 + 180.0 * pressure_quality,
        60.0,
        240.0,
    ))

    if score_details is not None:
        score_details.clear()
        score_details.update({
            "pressure_vertical_range": scoring.getPressureVerticalRange(
                pressure_relative_heights
            ),
            "pressure_vertical_score": pressure_vertical_score,
            "pressure_knee_range": scoring.getPressureKneeAngleRange(
                pressure_knee_angles
            ),
            "pressure_knee_range_score": pressure_knee_range_score,
            "rotation_separation_score": avg_rotation_separation_score,
            "rotation_direction_change_score": avg_skiAngle2_score,
            "rotation_lateral_score": recent_lateral_score,
            "edging_parallelism_score": avg_edging_angle_score,
            "edging_lateral_score": recent_lateral_score,
        })



    return Blue_edging_final,Blue_balance_final,Blue_rotation_final,Blue_pressure_final


import os

def slow_down_video_2x(video_path):
    """
    Slow down a video by 2x using FFmpeg in Colab.
    Overwrites the original file if successful.
    """

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # Check duration first
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video cannot be opened. Exiting.")
        return

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps if fps else 0
    cap.release()

    if duration == 0:
        print("Input video appears empty. Exiting.")
        return

    # Create temporary output path
    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_slow{ext}"

    # Check whether the video has audio
    probe = os.popen(f"ffprobe -i \"{video_path}\" -show_streams -select_streams a -loglevel error").read()
    has_audio = bool(probe.strip())

    if has_audio:
        # Video + audio slowdown
        cmd = f"""
        ffmpeg -y -i "{video_path}" \
        -filter_complex "[0:v]setpts=2.0*PTS[v];[0:a]atempo=0.5[a]" \
        -map "[v]" -map "[a]" "{temp_output}"
        """
    else:
        # Video only
        cmd = f"""
        ffmpeg -y -i "{video_path}" \
        -filter:v "setpts=2.0*PTS" "{temp_output}"
        """

    print("Running FFmpeg:")
    print(cmd)
    ret = os.system(cmd)

    if ret != 0:
        print("FFmpeg failed. Check video file.")
        return

    # Check if the temp output was created
    if not os.path.exists(temp_output):
        print("No output file was created. Exiting.")
        return

    # Replace the original video
    os.replace(temp_output, video_path)
    print(f"Video slowed down 2x and saved as: {video_path}")
