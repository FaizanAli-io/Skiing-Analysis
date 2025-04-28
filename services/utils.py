import cv2
import numpy as np
import os


def rotate_frame(frame, angle):
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


    def getLateralMovementScore(self,speed, max_score=150, min_score=0, min_speed=0, max_speed=40):
        if speed < 0:
            return 0

        if speed <= min_speed:
            return min_score
        elif speed >= max_speed:
            return max_score

        # Linear interpolation
        slope = (max_score - min_score) / (max_speed - min_speed)
        score = slope * (speed - min_speed) + min_score

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

    def getTiltScore(self,tilt_ratio, min_score=0, max_score=40):

        if not 0 <= tilt_ratio <= 1:
            raise ValueError("Tilt ratio must be between 0 and 1")

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

        print("SKI ANGLE 2: ")
        print(last_angles)
        print(total_change)
        print(avg_change)

        score = avg_change * 30

        score = min(score, 180)

        score = max(0, score)
        score = min(score, 180)
        return score

    def getBodyAngleScore(self, angle):
        print(f"Body Angle: {angle}")
        x = 180 / (45 - 85)
        score = 180 + x*(angle - 45)
        # score = 4 * (90 - angle)
        print(f"Body Angle Score: {score}")
        score = max(0, score)
        score = min(score, 180)
        return score

    def getKneeAngleScore(self, angle, skiAngle):
        if skiAngle > 20:
            return 0
        print(f"Knee angle: {angle}")
        score = 2.5 * (angle)
        print(f"Knee angle score: {score}")
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
                print(f"Started Turning at {angle} degrees - {time} seconds")
                start = time
            elif status == "Turning" and angle > stoppingAngle and time - start > 0.2:
                status = "Moving"
                print(f"Stopped Turning at {angle} degrees - {time} seconds")
                timeTaken = round(time - start, 1)
                start = None
                times.append(timeTaken)
                turn += 1
                print(f"Time Taken for turn {turn}: {timeTaken}")
        print(f"Turning times: {times}")
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
        print(avgTurningTime)
        # score = avgTurningTime * 250
        # print(score)
        # score = 210 - score
        print(score)
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


   # Draw keypoints
    cv2.circle(frame, (navel_x, navel_y), 5, (0, 255, 255), -1)
    cv2.circle(frame, (left_foot_x, left_foot_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (right_foot_x, right_foot_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (left_knee_x, left_knee_y), 5, (203, 192, 255), -1)
    cv2.circle(frame, (right_knee_x, right_knee_y), 5, (203, 192, 255), -1)
    cv2.circle(frame, (left_hip_x, left_hip_y), 5, (255, 0, 255), -1)
    cv2.circle(frame, (right_hip_x, right_hip_y), 5, (0, 255, 255), -1)

    # Draw lines
    cv2.line(frame, (navel_x, navel_y), (left_foot_x, left_foot_y), (255, 0, 0), 2)
    cv2.line(frame, (navel_x, navel_y), (right_foot_x, right_foot_y), (255, 0, 0), 2)
    cv2.line(frame, (navel_x, navel_y), (left_knee_x, left_knee_y), (203, 192, 255), 2)
    cv2.line(frame, (left_knee_x, left_knee_y), (left_foot_x, left_foot_y), (203, 192, 255), 2)
    cv2.line(frame, (navel_x, navel_y), (right_knee_x, right_knee_y), (203, 192, 255), 2)
    cv2.line(frame, (right_knee_x, right_knee_y), (right_foot_x, right_foot_y), (203, 192, 255), 2)

    cv2.line(frame, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (0, 255, 0), 2)
    cv2.line(frame, (left_hip_x, left_hip_y), (right_hip_x, right_hip_y), (255, 255, 0), 2)
    cv2.line(frame, (left_shoulder_x, left_shoulder_y), (left_hip_x, left_hip_y), (128, 0, 128), 2)
    cv2.line(frame, (right_shoulder_x, right_shoulder_y), (right_hip_x, right_hip_y), (128, 0, 128), 2)
    cv2.line(frame, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), (0, 255, 255), 2)
    cv2.line(frame, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), (0, 255, 255), 2)


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
        cv2.putText(frame, "Tilting", (125, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return 1
    else:
        cv2.putText(frame, "Not Tilting", (125, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return 0


def detect_ski_lines(frame, ski_boxes, flag):
    """
    Detects and draws ski lines from bounding boxes on the frame.

    Args:
        frame (np.ndarray): The image frame.
        ski_boxes (list of tuples): List of bounding boxes [(x1, y1, x2, y2), ...]

    Returns:
        list of tuples: List of detected ski lines [(x1, y1, x2, y2), ...]
    """
    ski_lines = []

    if ski_boxes:
        # Get the overall bounding box
        x_min = min([box[0] for box in ski_boxes])
        y_min = min([box[1] for box in ski_boxes])
        x_max = max([box[2] for box in ski_boxes])
        y_max = max([box[3] for box in ski_boxes])

        # Split the bounding box into two halves (left and right ski)
        mid_x = (x_min + x_max) // 2
        ski_boxes = [(x_min, y_min, mid_x, y_max), (mid_x, y_min, x_max, y_max)]

        if ski_boxes:
            flag = True

        for idx, (x1, y1, x2, y2) in enumerate(ski_boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
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

                ski_color = (255, 165, 0) if idx == 0 else (0, 165, 255)
                cv2.line(frame, (x1_l, y1_l), (x2_l, y2_l), ski_color, 4)

                ski_lines.append((x1_l, y1_l, x2_l, y2_l))

    return ski_lines,flag
