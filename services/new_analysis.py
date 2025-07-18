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
def create_aesthetic_overlay(frame, metrics, frame_number, TARGET_WIDTH):
    """Create professional-looking overlay with metrics and styling"""

    overlay = frame.copy()

    # Enhanced Colors
    bg_color = (30, 30, 30)
    light_blue = (219, 152, 52)
    #light_blue = (230, 216, 173)  # Light blue for headings (BGR)
    text_color = (255, 255, 255)  # White text
    green_color = (113, 204, 46)  # Low scores (good) - BGR
    blue_color = (219, 152, 52)   # Medium scores (better) - BGR
    red_color = (60, 76, 231)     # High scores (best) - BGR

    # Top Banner
    cv2.rectangle(overlay, (0, 0), (TARGET_WIDTH, 80), bg_color, -1)
    cv2.putText(overlay, "SKIING PERFORMANCE ANALYSIS", (50, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, light_blue, 2)
    cv2.putText(overlay, f"Frame: {frame_number}", (TARGET_WIDTH - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Right Panel (Final Scores)
    score_x = TARGET_WIDTH - 380  # Increased width for better spacing
    score_y = 100
    score_w = 360
    score_h = 270
    cv2.rectangle(overlay, (score_x, score_y),
                  (score_x + score_w, score_y + score_h), bg_color, -1)
    cv2.putText(overlay, "FINAL SCORES", (score_x + 20, score_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, light_blue, 2)

    # Fixed typo in variable name
    sum_scores = (metrics["Blue_pressure_final"] +
                  metrics["Blue_rotation_final"] +
                  metrics["Blue_edging_final"] +
                  metrics["Blue_balance_final"])
    blue_iq = sum_scores / 4

    final_scores = [
        ("Pressure", metrics["Blue_pressure_final"]),
        ("Rotation", metrics["Blue_rotation_final"]),
        ("Edging", metrics["Blue_edging_final"]),
        ("Balance", metrics["Blue_balance_final"]),
        ("BLUE IQ", blue_iq)
    ]

    bar_y_start = score_y + 60
    for i, (label, score) in enumerate(final_scores):
        y_pos = bar_y_start + i * 40  # Increased spacing between bars

        # Label with better spacing
        cv2.putText(overlay, label, (score_x + 20, y_pos + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2)  # Better font

        # Bar background with gap from label
        bar_start_x = score_x + 120  # Increased gap from label
        bar_bg_start = (bar_start_x, y_pos - 12)
        bar_bg_end = (bar_start_x + 180, y_pos + 12)
        cv2.rectangle(overlay, bar_bg_start, bar_bg_end, (60, 60, 60), -1)

        # Fill bar based on score
        fill_width = int((score / 240) * 180)
        bar_fill_end = (bar_start_x + fill_width, y_pos + 12)

        # Color based on score: Green (low/good) -> Blue (medium/better) -> Red (high/best)
        if score >= 180:
            bar_color = red_color      # Best performance
        elif score >= 120:
            bar_color = blue_color     # Better performance
        else:
            bar_color = green_color    # Good performance

        cv2.rectangle(overlay, bar_bg_start, bar_fill_end, bar_color, -1)

        # Score value with better spacing
        cv2.putText(overlay, f"{score:.0f}", (bar_start_x + 190, y_pos + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 2)

    # Real-time metrics panel (moved below Final Scores)
    panel_x = TARGET_WIDTH - 380  # Same x position as Final Scores
    panel_y = score_y + score_h + 20  # Position below Final Scores with 20px gap
    panel_w = 360  # Same width as Final Scores panel
    panel_h = 200
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), bg_color, -1)
    cv2.putText(overlay, "REAL-TIME METRICS", (panel_x + 20, panel_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, light_blue, 2)

    metrics_to_display = [
        ("Angle b/w the skies ", "ski_angle", ""),
        ("Angle b/w ski and vertical ", "ski_angle2", ""),
        ("Number of Turns ", "turns", "")
    ]

    y_offset = 70
    line_height = 35
    for label, key, unit in metrics_to_display:
        value = metrics[key]
        if isinstance(value, list):
            if value and value[-1] is not None:
                value_display = f"{value[-1]:.1f}{unit}"
            else:
                value_display = "N/A"
        else:
            if value is None:
                value_display = "N/A"
            else:
                value_display = f"{value}{unit}"

        # Example: color speed differently if over threshold
        if "Speed" in label and isinstance(value, (int, float)):
            color = green_color if value > 20 else red_color
        else:
            color = text_color

        cv2.putText(overlay, f"{label}: {value_display}",
                    (panel_x + 20, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height

    # Blend with frame
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

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
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

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
          frame = create_aesthetic_overlay(frame,metrices,index_counter, TARGET_WIDTH)

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
      cv2.destroyAllWindows()
      print(f"Duration: {duration}")
      print(f"Total Turns: {turns}")
      import time
      time.sleep(1)

      if os.path.exists(output_path):
          file_size = os.path.getsize(output_path)
          print(f"Video saved: {output_path} ({file_size} bytes)")
          slow_down_video_2x(output_path)
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
