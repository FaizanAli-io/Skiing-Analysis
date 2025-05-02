# person + ski + pose (final version) (scoring1)
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
# from google.colab.patches import cv2_imshow

from services.utils import *


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
    # output_path = "output_video.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

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
                    cv2.putText(frame, f"Speed : {speed:.2f} px/frame", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    speed_score = scoring.getLateralMovementScore(speed)

                    pressure_speed_score.append(speed_score)
                else:
                    cv2.putText(frame, "Speed : -- px/frame", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
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
                print(f"Angle between the skis: {skiAngle:.2f} degrees")

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
                        print(f"Estimated angle between the skis : {skiAngle:.2f} degrees")

            else:
                if skiAngles:
                    avg_count = min(len(skiAngles), 10)
                    if avg_count != 0:
                        skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                        skiAngles.append(skiAngle)
                        print(f"Estimated angle between the skis(no line) : {skiAngle:.2f} degrees")



        #handle case where red box not detecting, give avg of last 10 scores
        if not flag:
            if skiAngles:
                    avg_count = min(len(skiAngles), 10)
                    if avg_count != 0:
                        skiAngle = sum(skiAngles[-avg_count:]) / avg_count
                        skiAngles.append(skiAngle)
                        print(f"Estimated angle between the skis(no red box) : {skiAngle:.2f} degrees")
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


        # out.write(frame)

        #cv2.imshow("Frames : ",frame)

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
    cap.release()
    # out.release()
    print(f"Duration: {duration}")
    print(f"Total Turns: {turns}")
    if turns == 0:
        score_noOfTurns = 0
        score_turnTime = 0
    else:

        timePerTurn = duration/turns
        print(f"{timePerTurn} seconds per turn")
        print(f"{TskiAngles2} ANALYZE FOR TURN DURATION")
        turningTimes = scoring.getTurnDurations(TskiAngles2)
        turningTimes = remove_outliers(turningTimes)
        print(turningTimes)
        if turningTimes:
            avgTurningTime = round(sum(turningTimes) / len(turningTimes), 2)
            print(f"Average Turning Time: {avgTurningTime}")

            score_turnTime = scoring.getTurnTimeScore(avgTurningTime)
            score_turnTime = scoring.mapScore(score_turnTime)
            print(f"score_turnTime: {score_turnTime}")

            score_noOfTurns = scoring.getNoOfTurnsScore(timePerTurn)
            score_noOfTurns = scoring.mapScore(score_noOfTurns)
            print(f"score_noOfTurns: {score_noOfTurns}")




    #removing outliers


    edging_angle_score = remove_outliers(edging_angle_score)
    lateral_score = remove_outliers(lateral_score)
    bending_angle_score = remove_outliers(bending_angle_score)
    knee_angle_score = remove_outliers(knee_angle_score)
    skiAngle2_score = remove_outliers(skiAngle2_score)
    pressure_angle_score = remove_outliers(pressure_angle_score)
    pressure_speed_score = remove_outliers(pressure_speed_score)

    knee_angle_score = getMax50Percent(knee_angle_score)
    bending_angle_score = getMax50Percent(bending_angle_score)

    lateral_score = getMax50Percent(lateral_score)

    #Taking average

    avg_edging_angle_score = sum(edging_angle_score) / len(edging_angle_score)
    avg_lateral_score = sum(lateral_score) / len(lateral_score)
    avg_bending_angle_score = sum(bending_angle_score) / len(bending_angle_score)
    avg_knee_angle_score = sum(knee_angle_score) / len(knee_angle_score)
    avg_skiAngle2_score = sum(skiAngle2_score) / len(skiAngle2_score)
    avg_pressure_angle_score = sum(pressure_angle_score) / len(pressure_angle_score)
    avg_pressure_speed_score = sum(pressure_speed_score) / len(pressure_speed_score)
    avg_shoulder_hip_score = sum(hip_shoulder) / len(hip_shoulder)

    print(f"Avg hip-shoulder angle score : {avg_shoulder_hip_score}")
    #Edging scoring
    Blue_edging_score = 0.55*avg_edging_angle_score + 0.35*avg_lateral_score + 0.1*avg_shoulder_hip_score

    Blue_edging_final = Blue_edging_score

    #Balance scoring
    Blue_balance_score = 0.3*avg_bending_angle_score + 0.3*avg_knee_angle_score + 0.3*score_turnTime + 0.1*avg_shoulder_hip_score

    Blue_balance_final = Blue_balance_score

    #Rotation scoring
    Blue_rotation_score = (avg_skiAngle2_score + avg_lateral_score)/2

    Blue_rotation_final = Blue_rotation_score*0.7 + score_noOfTurns*0.3



    #Pressure scoring

    total_frames = scoring.getFramesInTurn(turns,target_fps)

    ratio = scoring.calculateTiltRatio(total_frames, count_tilt)

    tilt_score = scoring.getTiltScore(ratio)

    Blue_pressure_score = avg_pressure_angle_score + avg_pressure_speed_score + tilt_score

    Blue_pressure_final = scoring.mapScore(Blue_pressure_score)


    #printing
    print(f"Titl score : {tilt_score}")
    print(f"Pressure angle score : {avg_pressure_angle_score}")
    print(f"Blue_edging_final: {Blue_edging_final}")
    print(f"Blue_rotation_final: {Blue_rotation_final}")
    print(f"Blue_balance_final: {Blue_balance_final}")
    print(f"Blue_pressure_final: {Blue_pressure_final}")
    return {
        "pressure_score": Blue_pressure_final,
        "balance_score": Blue_balance_final,
        "rotation_score": Blue_rotation_final,
        "edging_score": Blue_edging_final
    }
