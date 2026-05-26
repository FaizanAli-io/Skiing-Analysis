# person + ski + pose (final version) (scoring1)
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ski_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Suppress ultralytics verbose logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

from services.new_utils import *

metrices = {
    "hip_angle" : [],
    "hip_vert_angle" : [],
    "tilt" :[],
    "speed/lateral movement" : [],
    "knee_angle" : [],
    "bend_angle" : [],
    "ski_angle" : [],
    "ski_angle2" : [],
    "turns" : [],
    "Blue_pressure_final": 0,
    "Blue_rotation_final": 0,
    "Blue_edging_final": 0,
    "Blue_balance_final": 0,
}

def draw_logo_placeholder(overlay, x, y, w, h, border_color, text_color):
    """Draw a placeholder when logo is not available"""
    logger.debug(f"Drawing logo placeholder at position ({x}, {y}) with dimensions {w}x{h}")
    try:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), border_color, 2)
        cv2.putText(overlay, "LOGO", (x + w//2 - 25, y + h//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        logger.debug("Logo placeholder drawn successfully")
    except Exception as e:
        logger.error(f"Error drawing logo placeholder: {e}")

def create_overlay(frame, metrics, frame_number, TARGET_WIDTH, logo_path=None):
    """Create professional-looking overlay with metrics and styling

    Args:
        frame: Video frame
        metrics: Performance metrics dictionary
        frame_number: Current frame number
        TARGET_WIDTH: Width of video frame
        logo_path: Path to logo image file (e.g., 'logo.png', 'company_logo.jpg')
    """
    logger.debug(f"Creating overlay for frame {frame_number}")
    
    try:
        # REVISED SIDEBAR DIMENSIONS - 650px width
        SIDEBAR_WIDTH = 720  # Significantly increased sidebar width
        EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
        frame_height = frame.shape[0]

        logger.debug(f"Frame dimensions: {frame_height}x{TARGET_WIDTH}, Extended width: {EXTENDED_WIDTH}")

        # Create extended frame with original video + white sidebar
        extended_frame = np.zeros((frame_height, EXTENDED_WIDTH, 3), dtype=np.uint8)
        extended_frame[:, :TARGET_WIDTH] = frame  # Original video on left
        # Set right side to white background
        extended_frame[:, TARGET_WIDTH:] = (255, 255, 255)  # White background for sidebar

        overlay = extended_frame.copy()

        # Professional Color Scheme
        header_bg_color = (255, 255, 255)      # White background for header
        white_bg = (255, 255, 255)            # White background
        primary_blue = (180, 100, 20)         # Professional blue for headings (BGR)
        text_dark = (50, 50, 50)              # Dark gray text for white background
        text_light = (255, 255, 255)          # White text for dark backgrounds

        # Ski slope colors (performance indicators)
        easy_green = (76, 175, 80)            # Green for beginner/good performance
        intermediate_blue = (180, 100, 20)    # Blue for intermediate performance
        expert_black = (50, 50, 50)           # Black for expert/challenging performance

        border_gray = (200, 200, 200)         # Light gray for borders
        grid_light = (230, 230, 230)          # Very light gray for grid lines
        separator_white = (255, 255, 255)     # White color for visible separators
        box_border = (180, 180, 180)          # Box container border color

        # CLEAN HEADER - No titles, just clean separation
        cv2.rectangle(overlay, (0, 0), (EXTENDED_WIDTH, 60), header_bg_color, -1)
        # cv2.line(overlay, (0, 102), (EXTENDED_WIDTH, 102), border_gray, 2)

        # Add clean separator line between video and sidebar
        cv2.line(overlay, (TARGET_WIDTH, 0), (TARGET_WIDTH, frame_height), border_gray, 3)

        # SIDEBAR CONTENT HIERARCHY - Starting from top
        sidebar_x = TARGET_WIDTH + 30  # Left margin in sidebar
        current_y = 80  # Start position after header

        # 1. LOGO - TOP & CENTERED IN SIDEBAR
        logo_w = 720
        logo_h = 150
        logo_x = TARGET_WIDTH + (SIDEBAR_WIDTH - logo_w) // 2  # Centered in 650px sidebar
        logo_y = current_y -80

        # Load and place logo image
        logo_loaded = False
        if logo_path and os.path.exists(logo_path):
            logger.debug(f"Attempting to load logo from: {logo_path}")
            try:
                logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

                if logo_img is not None:
                    logger.debug(f"Logo loaded successfully, shape: {logo_img.shape}")
                    if logo_img.shape[2] == 4:  # RGBA image
                        logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
                        alpha_channel = logo_resized[:, :, 3] / 255.0
                        logo_rgb = logo_resized[:, :, :3]

                        for c in range(3):
                            overlay[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w, c] = \
                                alpha_channel * logo_rgb[:, :, c] + \
                                (1 - alpha_channel) * overlay[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w, c]
                    else:
                        logo_resized = cv2.resize(logo_img, (logo_w, logo_h))
                        overlay[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w] = logo_resized

                    logo_loaded = True
                    logger.debug("Logo placed successfully on overlay")
                else:
                    logger.warning("Logo image could not be loaded - cv2.imread returned None")

            except Exception as e:
                logger.error(f"Error loading logo: {e}")
        else:
            if logo_path:
                logger.warning(f"Logo path provided but file does not exist: {logo_path}")
            else:
                logger.debug("No logo path provided")

        if not logo_loaded:
            logger.debug("Using logo placeholder")
            draw_logo_placeholder(overlay, logo_x, logo_y, logo_w, logo_h, border_gray, text_dark)

        current_y += logo_h - 30  # Move down after logo with padding

        # 2. FINAL SCORES BOX CONTAINER (SEPARATE BOX)
        box_padding = 20
        box_x = sidebar_x - 10
        box_width = 680

        # Calculate Final Scores box height based on content only
        scale_height = 100
        metrics_rows = 5  # 4 individual metrics + 1 overall
        generous_row_height = 55  # Generous vertical spacing
        final_scores_content_height = 40 + scale_height + (metrics_rows * generous_row_height) + 30
        final_scores_box_height = final_scores_content_height + (2 * box_padding)

        # Draw Final Scores box container
        cv2.rectangle(overlay, (box_x, current_y - 12),
                      (box_x + box_width, current_y + final_scores_box_height - 10),
                      (248, 248, 248), -1)  # Light gray background
        cv2.rectangle(overlay, (box_x, current_y - 12),
                      (box_x + box_width, current_y + final_scores_box_height - 10),
                      box_border, 2)  # Border

        # Section header inside Final Scores box
        cv2.putText(overlay, "ALPINE TRACK", (sidebar_x, current_y + 17),
                    cv2.FONT_HERSHEY_DUPLEX, 1, text_dark, 3)

        current_y += 50

        # SKILL LEVEL SCALE - ALIGNMENT REFERENCE SYSTEM
        scale_start_x = sidebar_x + 20  # Small left margin
        scale_width = 480  # Adjusted for box container
        scale_y = current_y
        scale_height = 50  # Optimized height for 720px constraint

        # CRITICAL: Define alignment coordinates that will be used by both scale and bars
        ALIGNMENT_SYSTEM = {
            'scale_start_x': scale_start_x + 95,
            'scale_width': scale_width,
            'min_score': 60,
            'max_score': 240,
            'beginner_end': 100,
            'intermediate_end': 200
        }

        # Function to calculate precise x-coordinate for any score value
        def get_x_for_score(score):
            score_range = ALIGNMENT_SYSTEM['max_score'] - ALIGNMENT_SYSTEM['min_score']
            position_ratio = (score - ALIGNMENT_SYSTEM['min_score']) / score_range
            return ALIGNMENT_SYSTEM['scale_start_x'] + int(position_ratio * ALIGNMENT_SYSTEM['scale_width'])

        # Scale values with PRECISE positioning using alignment system
        scale_values = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        scale_positions = []

        for val in scale_values:
            x_pos = get_x_for_score(val)
            scale_positions.append(x_pos)

            # Draw scale numbers with precise centering
            if val < 100:
                text_offset = 8   # Single/double digit numbers
            else:
                text_offset = 15  # Triple digit numbers

            cv2.putText(overlay, str(val), (x_pos - text_offset, scale_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_dark, 2)

            # Grid lines
            cv2.line(overlay, (x_pos, scale_y + 18), (x_pos, scale_y + scale_height - 5), grid_light, 1)

        # Skill level sections with PRECISE boundary positioning using alignment system
        section_height = 25  # Optimized for 720px height
        section_y = scale_y + 20

        # Calculate precise boundary positions using alignment system
        beginner_start = get_x_for_score(60)
        beginner_end = get_x_for_score(100)
        intermediate_end = get_x_for_score(200)
        expert_end = get_x_for_score(240)

        # Beginner section (60-99) - Green
        cv2.rectangle(overlay, (beginner_start, section_y),
                      (beginner_end, section_y + section_height), easy_green, -1)
        text_center_x = beginner_start + (beginner_end - beginner_start) // 2 - 40
        cv2.putText(overlay, "BEGINNER", (text_center_x + 1, section_y + 17),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(overlay, "BEGINNER", (text_center_x, section_y + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

        # Intermediate section (100-199) - Blue
        cv2.rectangle(overlay, (beginner_end, section_y),
                      (intermediate_end, section_y + section_height), intermediate_blue, -1)
        text_center_x = beginner_end + (intermediate_end - beginner_end) // 2 - 50
        cv2.putText(overlay, "INTERMEDIATE", (text_center_x + 1, section_y + 17),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(overlay, "INTERMEDIATE", (text_center_x, section_y + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

        # Expert section (200-240) - Black
        cv2.rectangle(overlay, (intermediate_end, section_y),
                      (expert_end, section_y + section_height), expert_black, -1)
        text_center_x = intermediate_end + (expert_end - intermediate_end) // 2 - 30
        cv2.putText(overlay, "EXPERT", (text_center_x + 1, section_y + 17),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(overlay, "EXPERT", (text_center_x, section_y + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

        # CRITICAL: VISIBLE separator lines using alignment system
        cv2.line(overlay, (beginner_end, section_y), (beginner_end, section_y + section_height),
                 separator_white, 3)
        cv2.line(overlay, (intermediate_end, section_y), (intermediate_end, section_y + section_height),
                 separator_white, 3)

        current_y += 80  # Move down after scale

        # Calculate scores
        sum_scores = (metrics["Blue_pressure_final"] +
                      metrics["Blue_rotation_final"] +
                      metrics["Blue_edging_final"] +
                      metrics["Blue_balance_final"])
        blue_iq = sum_scores / 4

        logger.debug(f"Calculated Blue IQ: {blue_iq:.2f} (sum: {sum_scores})")

        # SIMPLIFIED METRIC NAMES AND LAYOUT: Metric Name ------bar------- Score Value
        final_scores = [
            ("Balance", metrics["Blue_balance_final"]),
            ("Rotation", metrics["Blue_rotation_final"]),
            ("Pressure", metrics["Blue_pressure_final"]),
            ("Edging", metrics["Blue_edging_final"]),
            ("BLUE IQ", blue_iq)
        ]

        # INDIVIDUAL SCORE BARS - All aligned with scale using alignment system
        for i, (label, score) in enumerate(final_scores):
            y_pos = current_y + i * generous_row_height

            # Metric label - left aligned
            font_weight = 2
            label_color = primary_blue if "BLUE IQ" in label else text_dark
            font_size = 0.8

            cv2.putText(overlay, label, (sidebar_x, y_pos+9),
                        cv2.FONT_HERSHEY_DUPLEX, font_size, label_color, font_weight)

            # Score bar - positioned using EXACT same alignment system as scale
            bar_start_x = ALIGNMENT_SYSTEM['scale_start_x']  # EXACT same start as scale
            bar_width = ALIGNMENT_SYSTEM['scale_width']      # EXACT same width as scale
            bar_y = y_pos - 8
            bar_height = 20

            # Bar background
            cv2.rectangle(overlay, (bar_start_x, bar_y),
                         (bar_start_x + bar_width, bar_y + bar_height), (240, 240, 240), -1)
            cv2.rectangle(overlay, (bar_start_x, bar_y),
                         (bar_start_x + bar_width, bar_y + bar_height), (180, 180, 180), 1)

            # Bar fill - using alignment system for precise positioning
            if score >= 60:
                # Calculate fill position using alignment system
                fill_end_x = get_x_for_score(min(score, 240))
                fill_width = fill_end_x - bar_start_x

                # Color mapping
                if score >= 200:
                    bar_color = expert_black
                elif score >= 100:
                    bar_color = intermediate_blue
                else:
                    bar_color = easy_green

                # Fill bar
                cv2.rectangle(overlay, (bar_start_x, bar_y),
                             (fill_end_x, bar_y + bar_height), bar_color, -1)

                # Score marker
                cv2.circle(overlay, (fill_end_x, bar_y + bar_height//2), 4, (255, 255, 255), -1)
                cv2.circle(overlay, (fill_end_x, bar_y + bar_height//2), 4, text_dark, 1)

            # CRITICAL: Separator lines using EXACT same positions as scale
            beginner_separator_x = get_x_for_score(100)
            intermediate_separator_x = get_x_for_score(200)
            
            cv2.line(overlay, (beginner_separator_x, bar_y), (beginner_separator_x, bar_y + bar_height), 
                     separator_white, 2)
            cv2.line(overlay, (intermediate_separator_x, bar_y), (intermediate_separator_x, bar_y + bar_height), 
                     separator_white, 2)

            # Score value - positioned after the bar
            score_color = primary_blue if "BLUE IQ" in label else text_dark
            score_x = bar_start_x + bar_width + 15
            cv2.putText(overlay, f"{score:.0f}", (score_x, y_pos + 9),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, score_color, 2)

        current_y += len(final_scores) * generous_row_height - 3

        # ADD 5PX GAP BETWEEN BOXES
        current_y += 7

        # 3. REAL-TIME METRICS BOX CONTAINER (SEPARATE BOX)
        realtime_box_y = current_y
        realtime_box_height = 158  # Enough for 3 metrics with spacing

        # Draw separate box container for real-time metrics
        cv2.rectangle(overlay, (box_x, realtime_box_y - 10),
                      (box_x + box_width, realtime_box_y + realtime_box_height),
                      (248, 248, 248), -1)  # Light gray background
        cv2.rectangle(overlay, (box_x, realtime_box_y - 10),
                      (box_x + box_width, realtime_box_y + realtime_box_height),
                      box_border, 2)  # Border

        cv2.putText(overlay, "REAL-TIME METRICS", (sidebar_x, current_y + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, text_dark, 3)

        current_y += 70

        # Metrics display - optimized for remaining space
        metrics_to_display = [
            ("Ski Angle Separation", "ski_angle", "degree"),
            ("Ski-to-Vertical Angle", "ski_angle2", "degree"),
            ("Turn Count", "turns", "")
        ]

        line_height = 35  # Generous spacing for real-time metrics
        for label, key, unit in metrics_to_display:
            value = metrics[key]
            if isinstance(value, list):
                if value and value[-1] is not None:
                    value_display = f"   {value[-1]:.1f}  {unit} "
                else:
                    value_display = "  N/A"
            else:
                if value is None:
                    value_display = "  N/A"
                else:
                    value_display = f"  {value}  {unit}"

            # Display metric
            cv2.putText(overlay, f" {label} : ", (sidebar_x, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_dark, 2)
            cv2.putText(overlay, value_display, (sidebar_x + 250, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, primary_blue, 2)
            current_y += line_height

        # Professional blend - only sidebar area
        alpha = 0.95
        sidebar_region = extended_frame[:, TARGET_WIDTH:]
        overlay_sidebar = overlay[:, TARGET_WIDTH:]
        blended_sidebar = cv2.addWeighted(overlay_sidebar, alpha, sidebar_region, 1 - alpha, 0)
        extended_frame[:, TARGET_WIDTH:] = blended_sidebar

        logger.debug(f"Overlay created successfully for frame {frame_number}")
        return extended_frame

    except Exception as e:
        logger.error(f"Error creating overlay for frame {frame_number}: {e}")
        # Return original frame with sidebar if overlay creation fails
        try:
            SIDEBAR_WIDTH = 720
            EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
            frame_height = frame.shape[0]
            extended_frame = np.zeros((frame_height, EXTENDED_WIDTH, 3), dtype=np.uint8)
            extended_frame[:, :TARGET_WIDTH] = frame
            extended_frame[:, TARGET_WIDTH:] = (255, 255, 255)
            return extended_frame
        except:
            logger.critical("Failed to create fallback frame, returning original")
            return frame


def analyze_video(video_path: str) -> Dict[str, Any]:
    """Analyze ski video and return performance metrics"""
    logger.info(f"Starting video analysis for: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
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

        logger.info(f"Video properties - FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        logger.info(f"Processing every {frame_skip} frames (target FPS: {target_fps})")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        image_centre = frame_width // 2

        logger.info(f"Original frame dimensions: {frame_width}x{frame_height}")

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(base_dir, "outputs")

        # Ensure the outputs folder exists
        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(f"Output directory: {outputs_dir}")

        # Generate output path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(outputs_dir, f"output_{base_name}.mp4")
        logger.info(f"Output video path: {output_path}")

        out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH + 720, TARGET_HEIGHT))

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
        bendAngles = []
        bendAngle = None
        kneeAngles = []
        kneeAngle = None
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
                        
                        logger.info(f"Angle {angle:4d}°: nose_y={nose_y:.3f}, feet_y={feet_y:.3f}, score={score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            detected_angle = angle
                            logger.info(f"  ✓ Better orientation (nose is above feet)")
                    else:
                        logger.debug(f"Angle {angle}°: No pose landmarks detected")
            else:
                logger.warning("Could not read test frame, defaulting to 0° rotation")
                
        except Exception as e:
            logger.error(f"Error during rotation detection: {e}")
            detected_angle = 0
        
        # Close test capture and open fresh for main processing
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        logger.info(f"✓ Using rotation angle: {detected_angle}° (nose-above-feet score: {best_score:.3f})")
        frame_count = 0
        processed_frames = 0

        logger.info(f"Starting main video processing loop with rotation angle: {detected_angle}°")
        
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
                    logger.info(f"Frame shape after {detected_angle}° rotation: {frame.shape}")
            
            # STEP 2: Resize and center the rotated frame
            frame, new_width, new_height = resize_and_center_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
            image_centre = new_width // 2
            
            if frame_count == 0:
                logger.info(f"Frame shape after resize: {frame.shape} (width={new_width}, height={new_height})")
            
            # STEP 3: Skip frames if needed (for performance)
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_frames += 1
            if processed_frames % 50 == 0:
                logger.info(f"Processed {processed_frames} frames...")

            time += 1
            index_counter += 1
            
            try:
                results = model.track(frame, persist=True)
                
                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)

                detected_people = []
                ski_boxes = []

                detected_people, ski_boxes, detected_ids = detect_people_and_skis_from_results(results, scores, history)

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
                            logger.debug(f"Hip angle calculated: {hipAngle:.2f}°")

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
                if ski_boxes:
                    logger.debug(f"Processing {len(ski_boxes)} ski boxes")
                    try:
                        ski_lines, flag = detect_ski_lines(frame, ski_boxes, flag, detected_angle)
                        skiAngle = None

                        if len(ski_lines) == 2:
                            skiAngle = calculate_angle(ski_lines[0], ski_lines[1])
                            skiAngles.append(skiAngle)
                            logger.debug(f"Angle between skis: {skiAngle:.2f}°")

                            skiAngle2 = calculate_angle(ski_lines[0], (0, 0, 0, 1))
                            TskiAngles2.append((skiAngle2, time))

                        elif len(ski_lines) == 1:
                            skiAngle2 = calculate_angle(ski_lines[0], (0, 0, 0, 1))
                            TskiAngles2.append((skiAngle2, time))

                            if skiAngles:
                                avg_count = min(len(skiAngles), 10)
                                if avg_count != 0:
                                    skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                                    skiAngles.append(skiAngle)
                                    logger.debug(f"Estimated ski angle: {skiAngle:.2f}°")

                        else:
                            if skiAngles:
                                avg_count = min(len(skiAngles), 10)
                                if avg_count != 0:
                                    skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                                    skiAngles.append(skiAngle)

                    except Exception as e:
                        logger.warning(f"Error processing ski lines: {e}")

                # Handle case where ski detection fails
                if not flag:
                    if skiAngles:
                        avg_count = min(len(skiAngles), 10)
                        if avg_count != 0:
                            skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                            skiAngles.append(skiAngle)

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

                edging_angle_score.append(blue_score_skiAngle)
                lateral_score.append(blue_score_lateralMovement)
                bending_angle_score.append(blue_score_bendAngle)
                knee_angle_score.append(blue_score_kneeAngle)
                skiAngle2_score.append(blue_score_skiAngle2)

                # Update metrics
                metrices["ski_angle"].append(skiAngle)
                metrices["ski_angle2"].append(skiAngle2)
                metrices["turns"].append(turns)
                metrices["bend_angle"].append(bendAngle)
                metrices["knee_angle"].append(kneeAngle)
                metrices["hip_angle"].append(hipAngle)
                metrices["hip_vert_angle"].append(hipVertAngle)
                metrices["speed/lateral movement"].append(speed)
                metrices["tilt"].append(count_tilt)

                # Calculate final Blue scores
                Blue_edging_final, Blue_balance_final, Blue_rotation_final, Blue_pressure_final = calculate_blue_scores(
                    turns, duration, TskiAngles2, edging_angle_score, lateral_score, bending_angle_score,
                    knee_angle_score, skiAngle2_score, pressure_angle_score, pressure_speed_score,
                    hip_shoulder, scoring, target_fps, count_tilt
                )

                metrices["Blue_balance_final"] = Blue_balance_final
                metrices["Blue_edging_final"] = Blue_edging_final
                metrices["Blue_pressure_final"] = Blue_pressure_final
                metrices["Blue_rotation_final"] = Blue_rotation_final

                if processed_frames % 100 == 0:
                    logger.debug(f"Current scores - Edging: {Blue_edging_final:.1f}, "
                               f"Rotation: {Blue_rotation_final:.1f}, "
                               f"Balance: {Blue_balance_final:.1f}, "
                               f"Pressure: {Blue_pressure_final:.1f}")

                # Create overlay
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    logo_path = os.path.join(script_dir, "bluerun.png")
                    
                    frame = create_overlay(frame, metrices, index_counter, TARGET_WIDTH, logo_path)
                    out.write(frame)
                except Exception as e:
                    logger.error(f"Error creating overlay or writing frame: {e}")
                    # Write original frame if overlay fails
                    try:
                        SIDEBAR_WIDTH = 720
                        EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
                        frame_height = frame.shape[0]
                        extended_frame = np.zeros((frame_height, EXTENDED_WIDTH, 3), dtype=np.uint8)
                        extended_frame[:, :TARGET_WIDTH] = frame
                        extended_frame[:, TARGET_WIDTH:] = (255, 255, 255)
                        out.write(extended_frame)
                    except Exception as e2:
                        logger.error(f"Failed to write fallback frame: {e2}")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue

            frame_count += 1

        logger.info(f"Video processing completed. Total frames processed: {processed_frames}")

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise

    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        logger.info("Video capture and writer resources released")

    # Final calculations
    logger.info("Calculating final scores...")
    try:
        Blue_edging_final, Blue_balance_final, Blue_rotation_final, Blue_pressure_final = calculate_blue_scores(
            turns, duration, TskiAngles2, edging_angle_score, lateral_score, bending_angle_score,
            knee_angle_score, skiAngle2_score, pressure_angle_score, pressure_speed_score,
            hip_shoulder, scoring, target_fps, count_tilt
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
            "processed_frames": processed_frames
        }

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