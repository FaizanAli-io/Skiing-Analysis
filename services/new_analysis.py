# person + ski + pose (final version) (scoring1)
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import logging
import os
import time
from datetime import date
from typing import Dict, Any, Optional, Tuple, List

# Suppress verbose analysis messages. A dedicated logger below emits only the
# requested per-frame Pressure and Rotation details.
logger = logging.getLogger(__name__)
logger.handlers.clear()
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL + 1)

ANALYSIS_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ski_analysis.log",
)
frame_metrics_logger = logging.getLogger("blueiq.frame_metrics")
frame_metrics_logger.handlers.clear()
frame_metrics_logger.propagate = False
frame_metrics_logger.setLevel(logging.INFO)
frame_console_handler = logging.StreamHandler()
frame_console_handler.setFormatter(logging.Formatter("%(message)s"))
frame_metrics_logger.addHandler(frame_console_handler)


def _reset_frame_metrics_log():
    """Reuse and truncate ski_analysis.log for each video analysis."""
    for handler in list(frame_metrics_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            frame_metrics_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(
        ANALYSIS_LOG_PATH,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    frame_metrics_logger.addHandler(file_handler)

# Suppress ultralytics verbose logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

from services.new_utils import *

metrices = {
    "hip_angle" : [],
    "hip_vert_angle" : [],
    "tilt" :[],
    "speed/lateral movement" : [],
    "knee_angle" : [],
    "pressure_angle" : [],
    "pressure_knee_range" : [],
    "pressure_relative_height" : [],
    "pressure_vertical_range" : [],
    "rotation_angle" : [],
    "rotation_upper_angle" : [],
    "rotation_lower_angle" : [],
    "bend_angle" : [],
    "ski_angle" : [],
    "ski_angle2" : [],
    "edge_angle" : [],
    "turns" : [],
    "Blue_pressure_final": 0,
    "Blue_rotation_final": 0,
    "Blue_edging_final": 0,
    "Blue_balance_final": 0,
}

from services.overlay_renderers import create_overlay, create_premium_overlay, get_unique_output_path, write_react_overlay_page
from services.report_generator import generate_basic_report


def _safe_filename_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or "unknown"


def analyze_video(
    video_path: str,
    display_mode: str = "coach",
    overlay_renderer: str = "opencv",
    report: bool = False,
    user_name: Optional[str] = None,
    attempt_number: Optional[int] = None,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze ski video and return performance metrics"""
    _reset_frame_metrics_log()
    logger.info(f"Starting video analysis for: {video_path}")
    display_mode = display_mode.lower().strip()
    if display_mode not in {"coach", "athlete"}:
        logger.warning(f"Unknown display mode '{display_mode}', falling back to coach mode")
        display_mode = "coach"
    overlay_renderer = overlay_renderer.lower().strip()
    if overlay_renderer not in {"opencv", "react", "premium"}:
        logger.warning(f"Unknown overlay renderer '{overlay_renderer}', falling back to opencv")
        overlay_renderer = "opencv"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    session_date = session_date or date.today().isoformat()
    
    try:
        # Load YOLO model
        logger.info("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        logger.info("YOLO model loaded successfully")

        # Initialize MediaPipe Pose
        logger.info("Initializing MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        logger.info("MediaPipe Pose initialized successfully")

        TARGET_WIDTH = 1280
        TARGET_HEIGHT = 720

        # Load video
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError("Error: Could not open video file.")

        target_fps = 10
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if video_fps == 0:
            logger.error("Could not read FPS from the video")
            raise ValueError("Could not read FPS from the video. Please check the file format or path.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        frame_skip = max(1, video_fps // target_fps)
        output_fps = max(1.0, video_fps / frame_skip)

        logger.info(f"Video properties - FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        logger.info(f"Processing every {frame_skip} frames (target FPS: {target_fps})")
        logger.info(f"Output video FPS set to processed frame rate: {output_fps:.2f}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        image_centre = frame_width // 2

        logger.info(f"Original frame dimensions: {frame_width}x{frame_height}")

        # Use mp4v for OpenCV's writer (always software-based, no hardware encoder issues).
        # We transcode to browser-compatible H.264 (libx264) via ffmpeg after writing.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(base_dir, "outputs")

        # Ensure the outputs folder exists
        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(f"Output directory: {outputs_dir}")

        original_base_name = os.path.splitext(os.path.basename(video_path))[0]
        if user_name or attempt_number:
            context_parts = []
            if user_name:
                context_parts.append(_safe_filename_part(user_name))
            if attempt_number:
                context_parts.append(f"attempt_{attempt_number}")
            context_parts.append(_safe_filename_part(session_date))
            context_parts.append(_safe_filename_part(original_base_name))
            base_name = "_".join(context_parts)
        else:
            base_name = original_base_name
        output_variant = None if overlay_renderer == "opencv" else overlay_renderer
        output_path = get_unique_output_path(outputs_dir, base_name, display_mode, variant=output_variant)
        logger.info(f"Output video path: {output_path}")

        if overlay_renderer == "opencv":
            output_size = (TARGET_WIDTH + 720, TARGET_HEIGHT)
        elif overlay_renderer == "premium":
            output_size = (TARGET_WIDTH + 460, TARGET_HEIGHT)
        else:
            output_size = (TARGET_WIDTH, TARGET_HEIGHT)
        out = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)

        if not out.isOpened():
            logger.error("Failed to initialize video writer")
            raise ValueError("Failed to initialize video writer")

        # Initialize tracking and scoring variables
        logger.info("Initializing tracking and scoring variables...")
        track_memory = {"Person1": None, "Person2": None}
        last_seen = {"Person1": 0, "Person2": 0}
        frame_count = 0
        missing_threshold = 100

        scores = {}
        history = {}
        BLUEIQ_LEVELS = [
            (60, 80, "Beginner - Level 1"),
            (81, 100, "Beginner - Level 2"),
            (101, 120, "Intermediate - Level 3"),
            (121, 140, "Intermediate - Level 4"),
            (141, 160, "Intermediate - Level 5"),
            (161, 180, "Intermediate - Level 6"),
            (181, 200, "Intermediate/Expert - Level 7"),
            (201, 220, "Expert - Level 8"),
            (221, 240, "Expert - Level 9"),
        ]

        index_counter = 0
        hip_shoulder = []
        skiAngles = []
        skiAngle = None
        skiAngle2 = None
        TskiAngles2 = []
        edgeAngles = []
        edgeAngle = None
        bendAngles = []
        bendAngle = None
        kneeAngles = []
        kneeAngle = None
        pressureKneeAngles = []
        pressureKneeAngle = None
        pressureVerticalPositions = []
        pressureRelativeHeights = []
        pressureRelativeHeight = None
        rotationSeparationAngles = []
        rotationSeparationAngle = None
        rotationUpperAngle = None
        rotationLowerAngle = None
        hipAngles = []
        prev_point = None
        speed_list = []
        side = None
        turns = 0
        scoring = SkierScoring()

        # Initialize score variables
        score_skiAngle = 0
        score_skiAngle2 = 0
        score_bendAngle = 0
        score_kneeAngle = 0
        lateral_movement_score = 0
        time = 0
        count_tilt = 0
        actual_speed = []
        speed_score = 0

        # Initialize score lists
        edging_angle_score = []
        lateral_score = []
        bending_angle_score = []
        knee_angle_score = []
        skiAngle2_score = []
        pressure_speed_score = []
        pressure_angle_score = []
        rotation_separation_score = []
        score_timeline = []
        pillar_score_history = {
            "pressure": [],
            "balance": [],
            "rotation": [],
            "edging": [],
        }
        snapshot_path = os.path.splitext(output_path)[0] + "_snapshot.jpg"
        snapshot_saved = False
        snapshot_frame_target = max(1, int((total_frames / frame_skip) * 0.5))
        last_report_frame = None

        # Detect optimal rotation angle - person should be vertical/upright
        logger.info("Detecting optimal rotation angle using nose-above-feet test...")
        rotation_angles = [0, 90, -90, 180]
        detected_angle = 0  # Default: no rotation needed
        best_score = -999
        
        try:
            # Skip to middle of video for a cleaner frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, test_frame = cap.read()
            
            if ret:
                logger.info(f"Testing rotation on middle frame ({middle_frame}/{total_frames}) with shape: {test_frame.shape}")
                
                for angle in rotation_angles:
                    # Test the SAME frame at different angles
                    rotated_frame = rotate_frame(test_frame, angle)
                    resized_frame, new_width, new_height = resize_and_center_frame(rotated_frame, TARGET_WIDTH, TARGET_HEIGHT)
                    
                    # Check if person is upright using pose detection
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb_frame)
                    
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        
                        # Simple and reliable: nose should be ABOVE feet (smaller y = higher up)
                        nose_y = landmarks[0].y          # landmark 0 = nose
                        l_ankle_y = landmarks[27].y      # landmark 27 = left ankle
                        r_ankle_y = landmarks[28].y      # landmark 28 = right ankle
                        feet_y = (l_ankle_y + r_ankle_y) / 2
                        
                        # CORRECT orientation: nose_y < feet_y (nose is HIGHER = smaller y value)
                        # Score = how much ABOVE the feet the nose is
                        score = feet_y - nose_y
                        
                        logger.info(f"Angle {angle:4d}Â°: nose_y={nose_y:.3f}, feet_y={feet_y:.3f}, score={score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            detected_angle = angle
                            logger.info(f"  âœ“ Better orientation (nose is above feet)")
                    else:
                        logger.debug(f"Angle {angle}Â°: No pose landmarks detected")
            else:
                logger.warning("Could not read test frame, defaulting to 0Â° rotation")
                
        except Exception as e:
            logger.error(f"Error during rotation detection: {e}")
            detected_angle = 0
        
        # Close test capture and open fresh for main processing
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        logger.info(f"âœ“ Using rotation angle: {detected_angle}Â° (nose-above-feet score: {best_score:.3f})")
        frame_count = 0
        processed_frames = 0

        logger.info(f"Starting main video processing loop with rotation angle: {detected_angle}Â°")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached or error reading frame")
                break

            # Log original frame dimensions on first frame
            if frame_count == 0:
                logger.info(f"Original frame shape before rotation: {frame.shape}")

            # STEP 1: Rotate frame FIRST to make person upright (before any detection)
            if detected_angle != 0:
                frame = rotate_frame(frame, detected_angle)
                if frame_count == 0:
                    logger.info(f"Frame shape after {detected_angle}Â° rotation: {frame.shape}")
            
            # STEP 2: Resize and center the rotated frame
            frame, new_width, new_height = resize_and_center_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
            clean_display_frame = frame.copy()
            image_centre = new_width // 2
            
            if frame_count == 0:
                logger.info(f"Frame shape after resize: {frame.shape} (width={new_width}, height={new_height})")
            
            # STEP 3: Skip frames if needed (for performance)
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            try:
                # Reset frame-local measurements so a missed detection cannot
                # append stale values from the previous frame.
                hipAngle = None
                hipVertAngle = None
                speed = None
                kneeAngle = None
                bendAngle = None
                pressureKneeAngle = None
                pressureRelativeHeight = None
                skiAngle = None
                skiAngle2 = None
                edgeAngleMeasured = False

                results = model.track(frame, persist=True)
                
                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)

                detected_people = []
                ski_boxes = []
                biomech_body_points = None

                detected_people, ski_boxes, detected_ids = detect_people_and_skis_from_results(results, scores, history)

                pose_detected = pose_results.pose_landmarks is not None
                yolo_person_detected = bool(detected_people)

                # Skip only when both independent person signals fail.
                if not pose_detected and not yolo_person_detected:
                    frame_count += 1
                    continue

                # MediaPipe can retain a valid athlete pose when YOLO briefly
                # misses the person. Feed that pose through the existing single-
                # athlete path without inventing movement history for YOLO.
                if pose_detected and not yolo_person_detected:
                    fallback_track_id = track_memory["Person1"]
                    if fallback_track_id is None:
                        fallback_track_id = -1
                    track_memory["Person1"] = fallback_track_id
                    frame_h, frame_w = frame.shape[:2]
                    detected_people = [(
                        fallback_track_id,
                        0,
                        0,
                        frame_w,
                        frame_h,
                        frame_w,
                        frame_h,
                        frame_w // 2,
                        frame_h // 2,
                    )]

                processed_frames += 1
                time += 1
                index_counter += 1

                if processed_frames % 50 == 0:
                    logger.info(f"Processed {processed_frames} frames...")

                if detected_people:
                    # Sort to pick the most relevant person (e.g., tallest one)
                    detected_people.sort(key=lambda p: (-p[6], p[2]))
                    selected_person = detected_people[0]
                    track_id = selected_person[0]

                    track_memory["Person1"] = track_id
                    last_seen["Person1"] = frame_count

                for track_id in detected_ids:
                    side, turns = update_turns_and_side(track_id, history, image_centre, side, turns, scoring)

                for track_id, x1, y1, x2, y2, _, _, _, _ in detected_people:
                    label = "Unknown"
                    color = (255, 255, 255)

                    if track_id == track_memory["Person1"]:
                        label = "Person1"
                        color = (0, 255, 0)

                    for min_score, max_score, level in BLUEIQ_LEVELS:
                        if track_id in scores and min_score <= scores[track_id] <= max_score:
                            label += f" - {level} ({scores[track_id]})"
                            break

                    if pose_results.pose_landmarks and label != "Unknown":
                        logger.debug(f"Processing pose landmarks for {label}")
                        h, w, _ = frame.shape
                        landmarks = pose_results.pose_landmarks.landmark

                        # Extract key body points
                        try:
                            # Navel and legs
                            navel_x = int((landmarks[23].x + landmarks[24].x) / 2 * w)
                            navel_y = int((landmarks[23].y + landmarks[24].y) / 2 * h)
                            left_foot_x, left_foot_y = int(landmarks[27].x * w), int(landmarks[27].y * h)
                            right_foot_x, right_foot_y = int(landmarks[28].x * w), int(landmarks[28].y * h)
                            left_knee_x, left_knee_y = int(landmarks[25].x * w), int(landmarks[25].y * h)
                            right_knee_x, right_knee_y = int(landmarks[26].x * w), int(landmarks[26].y * h)

                            # Shoulders
                            left_shoulder_x, left_shoulder_y = int(landmarks[11].x * w), int(landmarks[11].y * h)
                            right_shoulder_x, right_shoulder_y = int(landmarks[12].x * w), int(landmarks[12].y * h)

                            # Hips
                            left_hip_x, left_hip_y = int(landmarks[23].x * w), int(landmarks[23].y * h)
                            right_hip_x, right_hip_y = int(landmarks[24].x * w), int(landmarks[24].y * h)
                            biomech_body_points = {
                                "navel": (navel_x, navel_y),
                                "left_foot": (left_foot_x, left_foot_y),
                                "right_foot": (right_foot_x, right_foot_y),
                                "left_knee": (left_knee_x, left_knee_y),
                                "right_knee": (right_knee_x, right_knee_y),
                                "left_shoulder": (left_shoulder_x, left_shoulder_y),
                                "right_shoulder": (right_shoulder_x, right_shoulder_y),
                                "left_hip": (left_hip_x, left_hip_y),
                                "right_hip": (right_hip_x, right_hip_y),
                            }

                            # Draw pose connections
                            frame = draw_pose_connections(frame, (navel_x, navel_y, left_foot_x, left_foot_y, 
                                                               right_foot_x, right_foot_y, left_knee_x, left_knee_y, 
                                                               right_knee_x, right_knee_y, left_shoulder_x, left_shoulder_y, 
                                                               right_shoulder_x, right_shoulder_y, left_hip_x, left_hip_y, 
                                                               right_hip_x, right_hip_y))

                            # Calculate angles
                            leftHipAngle = calculate_angle((left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y), 
                                                         (left_hip_x, left_hip_y, left_knee_x, left_knee_y))
                            rightHipAngle = calculate_angle((right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y), 
                                                          (right_hip_x, right_hip_y, right_knee_x, right_knee_y))

                            hipAngle = (leftHipAngle + rightHipAngle) / 2
                            logger.debug(f"Hip angle calculated: {hipAngle:.2f}Â°")

                            shoulder_hip_score = scoring.getShoulderHipScore(hipAngle)
                            hip_shoulder.append(shoulder_hip_score)
                            pressure_angle_score.append(scoring.getPressureAngleScore(hipAngle))
                            hipAngles.append(hipAngle)

                            leftHiptoVert = calculate_angle((left_knee_x, left_knee_y, left_foot_x, left_foot_y), (0, 0, 0, 1))
                            rightHiptoVert = calculate_angle((right_knee_x, right_knee_y, right_foot_x, right_foot_y), (0, 0, 0, 1))
                            hipVertAngle = (leftHiptoVert + rightHiptoVert) / 2

                            current_point = ((left_hip_x + right_hip_x) // 2, (left_hip_y + right_hip_y) // 2)

                            current_point, speed, speed_list, actual_speed = calculate_adjusted_speed(
                                (left_hip_x, left_hip_y), (right_hip_x, right_hip_y), prev_point, speed_list, actual_speed)

                            prev_point = current_point

                            if speed is not None:
                                if speed > 55:
                                    speed = 55
                                speed_score = scoring.getLateralMovementScore(speed, 100, 0, 0, 40)
                                logger.debug(f"Speed: {speed:.2f}, Speed Score: {speed_score}")
                                pressure_speed_score.append(speed_score)

                            tilting_flag = check_tilting(frame, speed, hipVertAngle)
                            if tilting_flag:
                                count_tilt += 1

                            leftKneeAngle = calculate_angle((navel_x, navel_y, left_knee_x, left_knee_y), 
                                                          (left_knee_x, left_knee_y, left_foot_x, left_foot_y))
                            rightKneeAngle = calculate_angle((navel_x, navel_y, right_knee_x, right_knee_y), 
                                                           (right_knee_x, right_knee_y, right_foot_x, right_foot_y))
                            kneeAngle = leftKneeAngle
                            if rightKneeAngle > kneeAngle:
                                kneeAngle = rightKneeAngle

                            kneeAngles.append(kneeAngle)

                            # Coach overlay Pressure measurement: true thigh-to-
                            # shin flexion at each knee. Keep it independent from
                            # the legacy navel-based knee score above.
                            leftPressureAngle = calculate_angle(
                                (left_knee_x, left_knee_y, left_hip_x, left_hip_y),
                                (left_knee_x, left_knee_y, left_foot_x, left_foot_y),
                            )
                            rightPressureAngle = calculate_angle(
                                (right_knee_x, right_knee_y, right_hip_x, right_hip_y),
                                (right_knee_x, right_knee_y, right_foot_x, right_foot_y),
                            )
                            pressureKneeAngle = (leftPressureAngle + rightPressureAngle) / 2
                            pressureKneeAngles.append(pressureKneeAngle)

                            hip_mid_y = (left_hip_y + right_hip_y) / 2
                            pressureVerticalPositions.append(hip_mid_y)

                            shoulder_mid = (
                                (left_shoulder_x + right_shoulder_x) / 2,
                                (left_shoulder_y + right_shoulder_y) / 2,
                            )
                            hip_mid = (
                                (left_hip_x + right_hip_x) / 2,
                                (left_hip_y + right_hip_y) / 2,
                            )
                            boot_mid = (
                                (left_foot_x + right_foot_x) / 2,
                                (left_foot_y + right_foot_y) / 2,
                            )
                            visible_body_height = float(np.hypot(
                                boot_mid[0] - shoulder_mid[0],
                                boot_mid[1] - shoulder_mid[1],
                            ))
                            if visible_body_height > 0:
                                pressureRelativeHeight = (
                                    boot_mid[1] - hip_mid[1]
                                ) / visible_body_height
                                pressureRelativeHeights.append(pressureRelativeHeight)

                            bendAngle1 = calculate_angle((navel_x, navel_y, left_foot_x, left_foot_y), (0, 0, 1, 0))
                            bendAngle2 = calculate_angle((navel_x, navel_y, right_foot_x, right_foot_y), (0, 0, 1, 0))
                            bendAngle = bendAngle1

                            if bendAngle2 < bendAngle:
                                bendAngle = bendAngle2

                            bendAngles.append(bendAngle)
                            score_bendAngle = scoring.getBodyAngleScore(bendAngle)

                        except Exception as e:
                            logger.warning(f"Error processing pose landmarks: {e}")
                            continue

                # Process ski detection
                flag = False
                ski_lines = []
                edgeAngle = None
                if ski_boxes:
                    logger.debug(f"Processing {len(ski_boxes)} ski boxes")
                    try:
                        ski_lines, flag = detect_ski_lines(frame, ski_boxes, flag, detected_angle)
                        skiAngle = None

                        if len(ski_lines) == 2:
                            skiAngle = calculate_angle(ski_lines[0], ski_lines[1])
                            skiAngles.append(skiAngle)
                            logger.debug(f"Angle between skis: {skiAngle:.2f}Â°")

                            # Average both ski bases against the horizontal
                            # carpet reference for the displayed edge angle.
                            detected_edge_angles = [
                                calculate_angle(line, (0, 0, 1, 0))
                                for line in ski_lines
                            ]
                            edgeAngle = sum(detected_edge_angles) / len(detected_edge_angles)
                            edgeAngles.append(edgeAngle)
                            edgeAngleMeasured = True

                            # Keep this legacy vertical-based value unchanged;
                            # existing rotation scoring and turn detection use it.
                            skiAngle2 = calculate_angle(ski_lines[0], (0, 0, 0, 1))
                            TskiAngles2.append((skiAngle2, time))

                        elif len(ski_lines) == 1:
                            skiAngle2 = calculate_angle(ski_lines[0], (0, 0, 0, 1))
                            TskiAngles2.append((skiAngle2, time))

                            # One ski is insufficient for a reliable edge angle.
                            if edgeAngles:
                                avg_count = min(len(edgeAngles), 10)
                                edgeAngle = sum(edgeAngles[-avg_count:]) / avg_count

                            if skiAngles:
                                avg_count = min(len(skiAngles), 10)
                                if avg_count != 0:
                                    skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                                    skiAngles.append(skiAngle)
                                    logger.debug(f"Estimated ski angle: {skiAngle:.2f}Â°")

                        else:
                            if edgeAngles:
                                avg_count = min(len(edgeAngles), 10)
                                edgeAngle = sum(edgeAngles[-avg_count:]) / avg_count
                            if skiAngles:
                                avg_count = min(len(skiAngles), 10)
                                if avg_count != 0:
                                    skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                                    skiAngles.append(skiAngle)

                    except Exception as e:
                        logger.warning(f"Error processing ski lines: {e}")

                # Handle case where ski detection fails
                if not flag:
                    if edgeAngles:
                        avg_count = min(len(edgeAngles), 10)
                        edgeAngle = sum(edgeAngles[-avg_count:]) / avg_count
                    if skiAngles:
                        avg_count = min(len(skiAngles), 10)
                        if avg_count != 0:
                            skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                            skiAngles.append(skiAngle)

                # Rotation proxy for a single front-facing camera. Compare the
                # signed lean of hip-to-shoulder and hip-to-boot midpoint lines.
                rotationSeparationAngle = None
                rotationUpperAngle = None
                rotationLowerAngle = None
                if biomech_body_points:
                    left_shoulder = biomech_body_points.get("left_shoulder")
                    right_shoulder = biomech_body_points.get("right_shoulder")
                    left_hip = biomech_body_points.get("left_hip")
                    right_hip = biomech_body_points.get("right_hip")
                    left_foot = biomech_body_points.get("left_foot")
                    right_foot = biomech_body_points.get("right_foot")
                    if all((left_shoulder, right_shoulder, left_hip, right_hip, left_foot, right_foot)):
                        shoulder_mid = (
                            (left_shoulder[0] + right_shoulder[0]) // 2,
                            (left_shoulder[1] + right_shoulder[1]) // 2,
                        )
                        hip_mid = (
                            (left_hip[0] + right_hip[0]) // 2,
                            (left_hip[1] + right_hip[1]) // 2,
                        )
                        boot_mid = (
                            (left_foot[0] + right_foot[0]) // 2,
                            (left_foot[1] + right_foot[1]) // 2,
                        )
                        rotationUpperAngle = calculate_signed_vertical_angle(
                            hip_mid, shoulder_mid, "up"
                        )
                        rotationLowerAngle = calculate_signed_vertical_angle(
                            hip_mid, boot_mid, "down"
                        )
                        raw_rotation_angle = abs(
                            (rotationLowerAngle - rotationUpperAngle + 180) % 360 - 180
                        )
                        rotationSeparationAngles.append(raw_rotation_angle)
                        avg_count = min(len(rotationSeparationAngles), 10)
                        rotationSeparationAngle = sum(rotationSeparationAngles[-avg_count:]) / avg_count
                        rotation_dynamics_score = scoring.getRotationDynamicsScore(
                            rotationSeparationAngles, target_fps
                        )
                        # Keep only the current whole-signal score. Averaging all
                        # earlier partial timelines would unfairly dilute it.
                        rotation_separation_score[:] = [rotation_dynamics_score]
                if rotationSeparationAngle is None and rotationSeparationAngles:
                    avg_count = min(len(rotationSeparationAngles), 10)
                    rotationSeparationAngle = sum(rotationSeparationAngles[-avg_count:]) / avg_count

                if display_mode == "coach" and biomech_body_points:
                    frame = draw_biomechanics_overlay(
                        frame,
                        biomech_body_points,
                        ski_lines,
                        {
                            "ski_separation": skiAngle,
                            "edge_angle": edgeAngle,
                            "pressure_angle": pressureKneeAngle,
                            "pressure_vertical_range": (
                                min(pressureRelativeHeights[-30:]),
                                max(pressureRelativeHeights[-30:]),
                                pressureRelativeHeights[-1],
                            ) if pressureRelativeHeights else None,
                            "rotation_angle": rotationSeparationAngle,
                            "rotation_upper_angle": rotationUpperAngle,
                            "rotation_lower_angle": rotationLowerAngle,
                            "hip_angle": hipAngle,
                            "bend_angle": bendAngle,
                        },
                    )

                # Calculate scores
                if skiAngle is not None:
                    score_skiAngle = scoring.getSkiAngleScore(skiAngle)
                    if kneeAngle is not None:
                        score_kneeAngle = scoring.getKneeAngleScore(kneeAngle, skiAngle)

                if skiAngle2 is not None and len(TskiAngles2) > 10:
                    score_skiAngle2 = scoring.getSkiAngle2Score(TskiAngles2)

                if speed is not None:
                    lateral_movement_score = scoring.getLateralMovementScore(speed, 180, 0, 0, 40)
                else:
                    lateral_movement_score = 0

                # Blue IQ Scoring
                blue_score_skiAngle = scoring.mapScore(score_skiAngle)
                blue_score_skiAngle2 = scoring.mapScore(score_skiAngle2)
                blue_score_bendAngle = scoring.mapScore(score_bendAngle)
                blue_score_kneeAngle = scoring.mapScore(score_kneeAngle)
                blue_score_lateralMovement = scoring.mapScore(lateral_movement_score)

                if skiAngle is not None:
                    edging_angle_score.append(blue_score_skiAngle)
                if speed is not None:
                    lateral_score.append(blue_score_lateralMovement)
                if bendAngle is not None:
                    bending_angle_score.append(blue_score_bendAngle)
                if kneeAngle is not None:
                    knee_angle_score.append(blue_score_kneeAngle)
                if skiAngle2 is not None and len(TskiAngles2) > 10:
                    skiAngle2_score.append(blue_score_skiAngle2)

                # Update metrics
                metrices["ski_angle"].append(skiAngle)
                metrices["ski_angle2"].append(skiAngle2)
                metrices["edge_angle"].append(edgeAngle)
                metrices["turns"].append(turns)
                metrices["bend_angle"].append(bendAngle)
                metrices["knee_angle"].append(kneeAngle)
                metrices["pressure_angle"].append(pressureKneeAngle)
                metrices["pressure_knee_range"].append(
                    scoring.getPressureKneeAngleRange(pressureKneeAngles)
                )
                metrices["pressure_relative_height"].append(pressureRelativeHeight)
                metrices["pressure_vertical_range"].append(
                    scoring.getPressureVerticalRange(pressureRelativeHeights)
                )
                metrices["rotation_angle"].append(rotationSeparationAngle)
                metrices["rotation_upper_angle"].append(rotationUpperAngle)
                metrices["rotation_lower_angle"].append(rotationLowerAngle)
                metrices["hip_angle"].append(hipAngle)
                metrices["hip_vert_angle"].append(hipVertAngle)
                metrices["speed/lateral movement"].append(speed)
                metrices["tilt"].append(count_tilt)

                # Calculate final Blue scores
                frame_score_details = {}
                frame_edging_score, frame_balance_score, frame_rotation_score, frame_pressure_score = calculate_blue_scores(
                    turns, duration, TskiAngles2, edging_angle_score, lateral_score, bending_angle_score,
                    knee_angle_score, skiAngle2_score, pressure_angle_score, pressure_speed_score,
                    pressureRelativeHeights, pressureKneeAngles,
                    rotation_separation_score, hip_shoulder,
                    scoring, target_fps, count_tilt,
                    score_details=frame_score_details,
                )

                frame_pillar_scores = {
                    "pressure": frame_pressure_score,
                    "balance": frame_balance_score,
                    "rotation": frame_rotation_score,
                    "edging": frame_edging_score,
                }
                for pillar, score in frame_pillar_scores.items():
                    if isinstance(score, (int, float)) and np.isfinite(score):
                        pillar_score_history[pillar].append(float(score))

                running_pillar_scores = {
                    pillar: sum(scores) / len(scores)
                    for pillar, scores in pillar_score_history.items()
                }
                Blue_pressure_final = running_pillar_scores["pressure"]
                Blue_balance_final = running_pillar_scores["balance"]
                Blue_rotation_final = running_pillar_scores["rotation"]
                Blue_edging_final = running_pillar_scores["edging"]

                metrices["Blue_balance_final"] = Blue_balance_final
                metrices["Blue_edging_final"] = Blue_edging_final
                metrices["Blue_pressure_final"] = Blue_pressure_final
                metrices["Blue_rotation_final"] = Blue_rotation_final

                pressure_degree_text = (
                    f"{pressureKneeAngle:.1f} deg"
                    if isinstance(pressureKneeAngle, (int, float))
                    else "N/A"
                )
                rotation_degree_text = (
                    f"{rotationSeparationAngle:.1f} deg"
                    if isinstance(rotationSeparationAngle, (int, float))
                    else "N/A"
                )
                separation_component = frame_score_details.get(
                    "rotation_separation_score"
                )
                separation_score_text = (
                    f"{separation_component:.1f}/240"
                    if isinstance(separation_component, (int, float))
                    else "N/A"
                )
                ski_parallelism_angle_text = (
                    f"{skiAngle:.1f} deg"
                    if isinstance(skiAngle, (int, float))
                    else "N/A"
                )
                edge_angle_text = (
                    f"{edgeAngle:.1f} deg"
                    if isinstance(edgeAngle, (int, float))
                    else "N/A"
                )
                edge_angle_source = (
                    "fresh two-ski detection"
                    if edgeAngleMeasured
                    else "recent fallback" if isinstance(edgeAngle, (int, float))
                    else "unavailable"
                )
                lateral_movement_text = (
                    f"{speed:.1f} px/frame"
                    if isinstance(speed, (int, float))
                    else "N/A"
                )
                edging_parallelism_component = frame_score_details[
                    "edging_parallelism_score"
                ]
                edging_lateral_component = frame_score_details[
                    "edging_lateral_score"
                ]
                frame_metrics_logger.info("\n".join([
                    "=" * 72,
                    f"FRAME {processed_frames}",
                    "=" * 72,
                    "PRESSURE",
                    f"  Current frame score        : {frame_pressure_score:.1f}/240",
                    f"  Running average (displayed): {Blue_pressure_final:.1f}/240",
                    f"  Current knee flexion       : {pressure_degree_text}",
                    "  Vertical movement range    : "
                    f"{frame_score_details['pressure_vertical_range'] * 100:.2f}% body height",
                    "  Vertical range score       : "
                    f"{frame_score_details['pressure_vertical_score']:.1f}/240 (75%)",
                    "  Knee flexion range         : "
                    f"{frame_score_details['pressure_knee_range']:.1f} deg",
                    "  Knee range score           : "
                    f"{frame_score_details['pressure_knee_range_score']:.1f}/240 (25%)",
                    "-" * 72,
                    "ROTATION",
                    f"  Current frame score        : {frame_rotation_score:.1f}/240",
                    f"  Running average (displayed): {Blue_rotation_final:.1f}/240",
                    f"  Body separation degree    : {rotation_degree_text}",
                    f"  Body separation score     : {separation_score_text} (70%)",
                    "  Direction-change score     : "
                    f"{frame_score_details['rotation_direction_change_score']:.1f}/240 (10%)",
                    "  Lateral-movement score     : "
                    f"{frame_score_details['rotation_lateral_score']:.1f}/240 (20%)",
                    "-" * 72,
                    "EDGING",
                    f"  Current frame score        : {frame_edging_score:.1f}/240",
                    f"  Running average (displayed): {Blue_edging_final:.1f}/240",
                    f"  Ski parallelism angle      : {ski_parallelism_angle_text}",
                    "  Parallelism component      : "
                    f"{edging_parallelism_component:.1f}/240 x 70% = "
                    f"{0.70 * edging_parallelism_component:.1f}",
                    f"  Edge angle ({edge_angle_source}) : {edge_angle_text}",
                    "  Edge-angle scoring         : ignored (display only)",
                    f"  Lateral movement           : {lateral_movement_text}",
                    "  Lateral component          : "
                    f"{edging_lateral_component:.1f}/240 x 30% = "
                    f"{0.30 * edging_lateral_component:.1f}",
                ]))

                score_timeline.append({
                    "time_seconds": round(processed_frames / output_fps, 2),
                    "pressure": frame_pressure_score,
                    "balance": frame_balance_score,
                    "rotation": frame_rotation_score,
                    "edging": frame_edging_score,
                    "ski_parallel_control": skiAngle,
                    "edge_control": edgeAngle,
                    "rotation_separation": rotationSeparationAngle,
                    "pressure_relative_height": pressureRelativeHeight,
                    "pressure_vertical_range": scoring.getPressureVerticalRange(
                        pressureRelativeHeights
                    ),
                    "pressure_knee_range": scoring.getPressureKneeAngleRange(
                        pressureKneeAngles
                    ),
                    "upper_body_alignment": hipAngle,
                    "athletic_stance_knee": kneeAngle,
                    "athletic_stance_bend": bendAngle,
                    "transition_control": speed,
                })

                if processed_frames % 100 == 0:
                    logger.debug(f"Current scores - Edging: {Blue_edging_final:.1f}, "
                               f"Rotation: {Blue_rotation_final:.1f}, "
                               f"Balance: {Blue_balance_final:.1f}, "
                               f"Pressure: {Blue_pressure_final:.1f}")

                # Create overlay
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    logo_path = os.path.join(script_dir, "bluerun.png")
                    
                    display_frame = frame if display_mode == "coach" else clean_display_frame
                    if overlay_renderer == "opencv":
                        frame = create_overlay(display_frame, metrices, index_counter, TARGET_WIDTH, logo_path, display_mode)
                        out.write(frame)
                    elif overlay_renderer == "premium":
                        frame = create_premium_overlay(
                            display_frame,
                            metrices,
                            index_counter,
                            TARGET_WIDTH,
                            logo_path,
                            display_mode,
                            athlete_name=user_name,
                            attempt_number=attempt_number,
                            session_date=session_date,
                        )
                        out.write(frame)
                    else:
                        out.write(display_frame)

                    if report:
                        last_report_frame = display_frame
                    if report and not snapshot_saved and processed_frames >= snapshot_frame_target:
                        if cv2.imwrite(snapshot_path, last_report_frame):
                            snapshot_saved = True
                            logger.info(f"Report technique snapshot saved: {snapshot_path}")
                        else:
                            logger.warning(f"Unable to save report technique snapshot: {snapshot_path}")
                except Exception as e:
                    logger.error(f"Error creating overlay or writing frame: {e}")
                    # Write original frame if overlay fails
                    try:
                        if overlay_renderer in {"opencv", "premium"}:
                            SIDEBAR_WIDTH = 460 if overlay_renderer == "premium" else 720
                            EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
                            frame_height = frame.shape[0]
                            extended_frame = np.zeros((frame_height, EXTENDED_WIDTH, 3), dtype=np.uint8)
                            extended_frame[:, :TARGET_WIDTH] = frame
                            extended_frame[:, TARGET_WIDTH:] = (255, 255, 255)
                            out.write(extended_frame)
                        else:
                            out.write(clean_display_frame if display_mode == "athlete" else frame)
                    except Exception as e2:
                        logger.error(f"Failed to write fallback frame: {e2}")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                frame_count += 1
                continue

            frame_count += 1

        logger.info(f"Video processing completed. Total frames processed: {processed_frames}")
        if report and not snapshot_saved and last_report_frame is not None:
            if cv2.imwrite(snapshot_path, last_report_frame):
                snapshot_saved = True
                logger.info(f"Fallback report technique snapshot saved: {snapshot_path}")
            else:
                logger.warning(f"Unable to save fallback report technique snapshot: {snapshot_path}")

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise

    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        logger.info("Video capture and writer resources released")
        
        # # Transcode to browser-compatible H.264 using libx264 explicitly,
        # # since OpenCV's own fourcc-based encoder selection is unreliable
        # # across platforms (e.g. picks unavailable hardware encoders on EC2).
        # if 'output_path' in locals() and os.path.exists(output_path):
        #     import subprocess
        #     temp_transcode_path = output_path + ".h264.mp4"
        #     try:
        #         subprocess.run(
        #             [
        #                 "ffmpeg", "-y",
        #                 "-i", output_path,
        #                 "-c:v", "libx264",
        #                 "-preset", "medium",
        #                 "-crf", "23",
        #                 "-pix_fmt", "yuv420p",
        #                 "-movflags", "+faststart",
        #                 temp_transcode_path,
        #             ],
        #             check=True,
        #             capture_output=True,
        #             text=True,
        #         )
        #         os.replace(temp_transcode_path, output_path)
        #         logger.info(f"Transcoded output video to H.264: {output_path}")
        #     except subprocess.CalledProcessError as e:
        #         logger.error(f"ffmpeg transcode failed, keeping mp4v output: {e.stderr}")
        #     except FileNotFoundError:
        #         logger.error("ffmpeg binary not found on PATH, keeping mp4v output")
        # Transcode to browser-compatible H.264 using libx264 explicitly,
        # since OpenCV's own fourcc-based encoder selection is unreliable
        # across platforms (e.g. picks unavailable hardware encoders on EC2).
        if 'output_path' in locals() and os.path.exists(output_path):
            import subprocess
            import shutil as _shutil

            ffmpeg_bin = _shutil.which("ffmpeg")
            if not ffmpeg_bin:
                for candidate in (
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    "/usr/bin/ffmpeg",
                    "/usr/local/bin/ffmpeg",
                ):
                    if os.path.exists(candidate):
                        ffmpeg_bin = candidate
                        break

            if not ffmpeg_bin:
                logger.error(
                    "ffmpeg not found on PATH or in known install locations. "
                    "Output remains mp4v and will NOT play in browsers."
                )
                raise RuntimeError(
                    "ffmpeg is required to produce a browser-playable video but was not found on this machine."
                )

            temp_transcode_path = output_path + ".h264.mp4"
            try:
                proc = subprocess.run(
                    [
                        ffmpeg_bin, "-y",
                        "-i", output_path,
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        temp_transcode_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if not os.path.exists(temp_transcode_path) or os.path.getsize(temp_transcode_path) == 0:
                    raise RuntimeError(f"ffmpeg produced no usable output. stderr: {proc.stderr}")
                os.replace(temp_transcode_path, output_path)
                logger.info(f"Transcoded output video to H.264 using: {ffmpeg_bin}")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg transcode failed (exit {e.returncode}): {e.stderr}")
                raise RuntimeError(f"ffmpeg transcode failed: {e.stderr}") from e
    # Final calculations
    logger.info("Calculating final scores...")
    try:
        if not pillar_score_history["pressure"]:
            raise ValueError("No valid frame scores were produced for this video")

        Blue_pressure_final = sum(pillar_score_history["pressure"]) / len(
            pillar_score_history["pressure"]
        )
        Blue_balance_final = sum(pillar_score_history["balance"]) / len(
            pillar_score_history["balance"]
        )
        Blue_rotation_final = sum(pillar_score_history["rotation"]) / len(
            pillar_score_history["rotation"]
        )
        Blue_edging_final = sum(pillar_score_history["edging"]) / len(
            pillar_score_history["edging"]
        )

        logger.info(f"Final Scores:")
        logger.info(f"  Blue_edging_final: {Blue_edging_final:.2f}")
        logger.info(f"  Blue_rotation_final: {Blue_rotation_final:.2f}")
        logger.info(f"  Blue_balance_final: {Blue_balance_final:.2f}")
        logger.info(f"  Blue_pressure_final: {Blue_pressure_final:.2f}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Total Turns: {turns}")

        metrices["Blue_balance_final"] = Blue_balance_final
        metrices["Blue_edging_final"] = Blue_edging_final
        metrices["Blue_pressure_final"] = Blue_pressure_final
        metrices["Blue_rotation_final"] = Blue_rotation_final

        # Check output file
        # time.sleep(1)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Video saved successfully: {output_path} ({file_size} bytes)")
        else:
            logger.warning(f"Output video file not found: {output_path}")

        result = {
            "pressure_score": Blue_pressure_final,
            "balance_score": Blue_balance_final,
            "rotation_score": Blue_rotation_final,
            "edging_score": Blue_edging_final,
            "output_path": output_path,
            "duration": duration,
            "turns": turns,
            "processed_frames": processed_frames,
            "display_mode": display_mode,
            "overlay_renderer": overlay_renderer,
            "user_name": user_name,
            "attempt_number": attempt_number,
            "session_date": session_date,
        }
        if 'snapshot_path' in locals() and snapshot_saved and os.path.exists(snapshot_path):
            result["snapshot_path"] = snapshot_path
        if overlay_renderer == "react":
            react_overlay_path = write_react_overlay_page(output_path, result, display_mode)
            result["react_overlay_path"] = react_overlay_path
        if report:
            report_result = generate_basic_report(
                result,
                score_timeline,
                context={
                    "user_name": user_name,
                    "attempt_number": attempt_number,
                    "session_date": session_date,
                },
            )
            result["report_text"] = report_result["report_text"]
            result["report_path"] = report_result["report_path"]
            result["score_windows"] = report_result["score_windows"]

        logger.info("Video analysis completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error calculating final scores: {e}")
        raise


if __name__ == "__main__":
    # Example usage with logging
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        logger.info(f"Starting analysis with video: {video_path}")
        try:
            results = analyze_video(video_path)
            logger.info(f"Analysis results: {results}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            sys.exit(1)
    else:
        logger.warning("No video path provided as command line argument")
        print("Usage: python ski_analysis.py <video_path>")

