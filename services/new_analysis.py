# person + ski + pose (final version) (scoring1)
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

#from google.colab.patches import cv2_imshow

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
    cv2.rectangle(overlay, (x, y), (x + w, y + h), border_color, 2)
    cv2.putText(overlay, "LOGO", (x + w//2 - 25, y + h//2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

def create_overlay(frame, metrics, frame_number, TARGET_WIDTH, logo_path=None):
    """Create professional-looking overlay with metrics and styling

    Args:
        frame: Video frame
        metrics: Performance metrics dictionary
        frame_number: Current frame number
        TARGET_WIDTH: Width of video frame
        logo_path: Path to logo image file (e.g., 'logo.png', 'company_logo.jpg')
    """

    # REVISED SIDEBAR DIMENSIONS - 650px width
    SIDEBAR_WIDTH = 720  # Significantly increased sidebar width
    EXTENDED_WIDTH = TARGET_WIDTH + SIDEBAR_WIDTH
    frame_height = frame.shape[0]

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
    cv2.line(overlay, (0, 102), (EXTENDED_WIDTH, 102), border_gray, 2)

    # Add clean separator line between video and sidebar
    cv2.line(overlay, (TARGET_WIDTH, 0), (TARGET_WIDTH, frame_height), border_gray, 3)

    # SIDEBAR CONTENT HIERARCHY - Starting from top
    sidebar_x = TARGET_WIDTH + 30  # Left margin in sidebar
    current_y = 80  # Start position after header

    # 1. LOGO - TOP & CENTERED IN SIDEBAR
    logo_w = 720
    logo_h = 100
    logo_x = TARGET_WIDTH + (SIDEBAR_WIDTH - logo_w) // 2  # Centered in 650px sidebar
    logo_y = current_y -80

    # Load and place logo image
    logo_loaded = False
    if logo_path and os.path.exists(logo_path):
        try:
            logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

            if logo_img is not None:
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

        except Exception as e:
            print(f"Error loading logo: {e}")

    if not logo_loaded:
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

    # Enhanced scale background
    # cv2.rectangle(overlay, (scale_start_x - 10 + 95, scale_y - 5),
    #               (scale_start_x + scale_width + 25 + 95, scale_y + scale_height + 5), text_dark, 2)
    # cv2.rectangle(overlay, (scale_start_x - 8 + 95, scale_y - 3),
    #               (scale_start_x + scale_width + 23 + 95, scale_y + scale_height + 3), (248, 248, 248), -1)

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

    return extended_frame


def analyze_video(video_path: str):

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()


    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720

    # Load video
    cap = cv2.VideoCapture(video_path)

    target_fps = 10
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if video_fps == 0:
        raise ValueError("Could not read FPS from the video. Please check the file format or path.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
    duration = total_frames / video_fps
    frame_skip = max(1, video_fps // target_fps)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the original FPS
    image_centre = frame_width//2

    # Define the codec and create a VideoWriter object

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec format

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go one level up
    outputs_dir = os.path.join(base_dir, "outputs")

    # Ensure the outputs folder exists
    os.makedirs(outputs_dir, exist_ok=True)

    # Generate output path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(outputs_dir, f"output_{base_name}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH + 720, TARGET_HEIGHT))

    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")

    # Tracking memory
    track_memory = {"Person1": None, "Person2": None}
    last_seen = {"Person1": 0, "Person2": 0}
    frame_count = 0
    missing_threshold = 100

    # Scoring memory
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

    hipAngles=[]



    prev_point = None
    speed_list = []


    side = None
    turns = 0
    scoring = SkierScoring()



    score_skiAngle = 0
    score_skiAngle2 = 0
    score_bendAngle = 0
    score_kneeAngle = 0
    lateral_movement_score = 0



    time = 0
    count_tilt = 0

    actual_speed = []

    speed_score = 0

    edging_angle_score = []
    lateral_score = []
    bending_angle_score = []
    knee_angle_score = []
    skiAngle2_score = []
    pressure_speed_score = []
    pressure_angle_score = []

    rotation_angles = [0, 90, -90, 180]
    detected_angle = None
    try:
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              print("No more frames or error reading frame.")
              break

          for angle in rotation_angles:
              frame = rotate_frame(frame, angle)
              frame, new_width, new_height = resize_and_center_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)

              image_centre = new_width //2
              if frame_count % frame_skip != 0:
                  frame_count += 1
                  continue


              results = model.track(frame, persist=True)

              # Convert frame to RGB for MediaPipe
              rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              pose_results = pose.process(rgb_frame)

              detected_people = []
              ski_boxes = []
              check = False


              detected_people, ski_boxes, detected_ids = detect_people_and_skis_from_results(results, scores, history)
              if detected_ids:
                  detected_angle = angle
                  print(f"People detected at {angle}° rotation")
                  check = True
                  break
          if check:
              break

      cap.release()
      if detected_angle is None:
          print("No people detected in any orientation.")
      else:
          print(f"Detected angle : {detected_angle}")
      cap = cv2.VideoCapture(video_path)
      while cap.isOpened():

          ret, frame = cap.read()
          if not ret:
              break

          frame, new_width, new_height = resize_and_center_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)

          image_centre = new_width//2
          if frame_count % frame_skip != 0:
              frame_count += 1
              continue

          time += 1
          index_counter += 1
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
                  if min_score <= scores[track_id] <= max_score:
                      label += f" - {level} ({scores[track_id]})"
                      break

              if pose_results.pose_landmarks and label != "Unknown":
                  h, w, _ = frame.shape
                  landmarks = pose_results.pose_landmarks.landmark

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

                  # Draw points

                  frame = draw_pose_connections(frame, (navel_x,navel_y, left_foot_x,left_foot_y, right_foot_x,right_foot_y,  left_knee_x, left_knee_y,  right_knee_x, right_knee_y,left_shoulder_x, left_shoulder_y,  right_shoulder_x, right_shoulder_y, left_hip_x, left_hip_y, right_hip_x, right_hip_y))



                  leftHipAngle = calculate_angle((left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y), (left_hip_x, left_hip_y, left_knee_x, left_knee_y))
                  rightHipAngle = calculate_angle((right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y), (right_hip_x, right_hip_y, right_knee_x, right_knee_y))


                  hipAngle = (leftHipAngle + rightHipAngle)/2


                  shoulder_hip_score = scoring.getShoulderHipScore(hipAngle)
                  hip_shoulder.append(shoulder_hip_score)

                  pressure_angle_score.append(scoring.getPressureAngleScore(hipAngle))
                  hipAngles.append(hipAngle)

                  leftHiptoVert = calculate_angle((left_knee_x, left_knee_y, left_foot_x, left_foot_y), (0,0,0,1))
                  rightHiptoVert = calculate_angle((right_knee_x, right_knee_y, right_foot_x, right_foot_y), (0,0,0,1))
                  hipVertAngle = (leftHiptoVert + rightHiptoVert)/2


                  current_point = ((left_hip_x + right_hip_x) // 2, (left_hip_y + right_hip_y) // 2)

                  current_point, speed, speed_list, actual_speed = calculate_adjusted_speed((left_hip_x, left_hip_y),(right_hip_x, right_hip_y),prev_point,speed_list,actual_speed)

                  prev_point = current_point

                  if speed is not None:

                      # cv2.putText(frame, f"Speed : {speed:.2f} px/frame", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                      if speed>55:
                        speed = 55
                      speed_score = scoring.getLateralMovementScore(speed,100,0,0,40)
                      print(f"Speed : {speed}, Speed Score: {speed_score}")

                      pressure_speed_score.append(speed_score)
                  # else:
                      # cv2.putText(frame, "Speed : -- px/frame", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                  tilting_flag = check_tilting(frame, speed, hipVertAngle)
                  if tilting_flag:
                      count_tilt += 1

                  leftKneeAngle = calculate_angle((navel_x, navel_y, left_knee_x, left_knee_y), (left_knee_x, left_knee_y,left_foot_x, left_foot_y))
                  rightKneeAngle = calculate_angle((navel_x, navel_y, right_knee_x, right_knee_y), (right_knee_x, right_knee_y,right_foot_x, right_foot_y))
                  kneeAngle = leftKneeAngle
                  if rightKneeAngle > kneeAngle:
                      kneeAngle =  rightKneeAngle

                  kneeAngles.append(kneeAngle)

                  bendAngle1 = calculate_angle((navel_x, navel_y, left_foot_x, left_foot_y), (0,0,1,0))
                  bendAngle2 = calculate_angle((navel_x, navel_y, right_foot_x, right_foot_y), (0,0,1,0))
                  bendAngle = bendAngle1

                  if bendAngle2 < bendAngle:
                      bendAngle = bendAngle2

                  bendAngles.append(bendAngle)

                  score_bendAngle = scoring.getBodyAngleScore(bendAngle)

          flag = False
          if ski_boxes:
              ski_lines,flag = detect_ski_lines(frame, ski_boxes, flag)
              skiAngle = None

              if len(ski_lines) == 2:
                  skiAngle = calculate_angle(ski_lines[0], ski_lines[1])
                  skiAngles.append(skiAngle)
                  # print(f"Angle between the skis: {skiAngle:.2f} degrees")

                  skiAngle2 = calculate_angle(ski_lines[0], (0,0,0,1))

                  TskiAngles2.append((skiAngle2, time))


              elif len(ski_lines) == 1:
                  skiAngle2 = calculate_angle(ski_lines[0], (0,0,0,1))

                  TskiAngles2.append((skiAngle2, time))

                  if skiAngles:
                      avg_count = min(len(skiAngles), 10)
                      if avg_count != 0:
                          skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                          skiAngles.append(skiAngle)
                          # print(f"Estimated angle between the skis : {skiAngle:.2f} degrees")

              else:
                  if skiAngles:
                      avg_count = min(len(skiAngles), 10)
                      if avg_count != 0:
                          skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                          skiAngles.append(skiAngle)
                          # print(f"Estimated angle between the skis(no line) : {skiAngle:.2f} degrees")



          #handle case where red box not detecting, give avg of last 10 scores
          if not flag:
              if skiAngles:
                      avg_count = min(len(skiAngles), 10)
                      if avg_count != 0:
                          skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                          skiAngles.append(skiAngle)
                          # print(f"Estimated angle between the skis(no red box) : {skiAngle:.2f} degrees")
          if skiAngle is not None:
              score_skiAngle = scoring.getSkiAngleScore(skiAngle)

              if kneeAngle is not None:
                  score_kneeAngle = scoring.getKneeAngleScore(kneeAngle, skiAngle)

          if skiAngle2 is not None and len(TskiAngles2)>10:
              score_skiAngle2 = scoring.getSkiAngle2Score(TskiAngles2)

          if speed is not None:
              lateral_movement_score = scoring.getLateralMovementScore(speed, 180, 0, 0, 40)
          else:
              lateral_movement_score = 0



          #print("Blue IQ Scoring")
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

          #metrices
          metrices["ski_angle"].append(skiAngle)
          metrices["ski_angle2"].append(skiAngle2)
          metrices["turns"].append(turns)
          metrices["bend_angle"].append(bendAngle)
          metrices["knee_angle"].append(kneeAngle)
          metrices["hip_angle"].append(hipAngle)
          metrices["hip_vert_angle"].append(hipVertAngle)
          metrices["speed/lateral movement"].append(speed)
          metrices["tilt"].append(count_tilt)


          Blue_edging_final,Blue_balance_final,Blue_rotation_final,Blue_pressure_final = calculate_blue_scores(
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
          hip_shoulder,
          scoring,
          target_fps,
          count_tilt
          )
          #printing

          metrices["Blue_balance_final"] = Blue_balance_final
          metrices["Blue_edging_final"] = Blue_edging_final
          metrices["Blue_pressure_final"] = Blue_pressure_final
          metrices["Blue_rotation_final"] = Blue_rotation_final

          print(f"Blue_edging_final: {Blue_edging_final}")
          print(f"Blue_rotation_final: {Blue_rotation_final}")
          print(f"Blue_balance_final: {Blue_balance_final}")
          print(f"Blue_pressure_final: {Blue_pressure_final}")

          #create
          script_dir = os.path.dirname(os.path.abspath(__file__))
          logo_path = os.path.join(script_dir, "bluerun.png")
          
          frame = create_overlay(frame,metrices,index_counter, TARGET_WIDTH,logo_path)

          out.write(frame)

          # cv2_imshow(frame)

          # key = cv2.waitKey(5) & 0xFF  # Wait for a key press for 5 milliseconds
          # if key == ord('q'):  # Press 'q' to quit
          #     break
          # elif key == ord(' '):  # Press spacebar to pause
          #     print("Paused. Press any key to resume...")
          #     while True:
          #         # Wait indefinitely for another key press
          #         if cv2.waitKey(0) & 0xFF:
          #             break  # Resume when any key is pressedk
          frame_count += 1

    except Exception as e:
        print(f"Error during video processing: {e}")

    finally:
      cap.release()
      out.release()
    #   cv2.destroyAllWindows()
      print(f"Duration: {duration}")
      print(f"Total Turns: {turns}")
      print(f"-------------------------------------------------------------------------------------------------------------")
      print(f"-------------------------------------------------------------------------------------------------------------")
      print(f"-------------------------------------------------------------------------------------------------------------")
      print(f"-------------------------------------------------------------------------------------------------------------")
      import time
      time.sleep(1)

      if os.path.exists(output_path):
          file_size = os.path.getsize(output_path)
          print(f"Video saved: {output_path} ({file_size} bytes)")
        #   slow_down_video_2x(output_path)
      else:
          print(f"Warning: Video file not created at {output_path}")

      Blue_edging_final,Blue_balance_final,Blue_rotation_final,Blue_pressure_final = calculate_blue_scores(
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
      hip_shoulder,
      scoring,
      target_fps,
      count_tilt
  )
      #printing

      print(f"Blue_edging_final: {Blue_edging_final}")
      print(f"Blue_rotation_final: {Blue_rotation_final}")
      print(f"Blue_balance_final: {Blue_balance_final}")
      print(f"Blue_pressure_final: {Blue_pressure_final}")
      metrices["Blue_balance_final"] = Blue_balance_final
      metrices["Blue_edging_final"] = Blue_edging_final
      metrices["Blue_pressure_final"] = Blue_pressure_final
      metrices["Blue_rotation_final"] = Blue_rotation_final
      return {
          "pressure_score": Blue_pressure_final,
          "balance_score": Blue_balance_final,
          "rotation_score": Blue_rotation_final,
          "edging_score": Blue_edging_final

      }
