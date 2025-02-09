import cv2
import mediapipe as mp
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import board
import busio
import time

try:
    # Initialize I2C bus and PCA9685
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 50

    # Create servo objects for each DOF
    servos = {
        "left_right": servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400),
        "up_down": servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400),
        "claw": servo.Servo(pca.channels[2], min_pulse=600, max_pulse=2400)
    }
except Exception as e:
    print(f"Error initializing servos: {e}")
    exit(1)

# Balanced MediaPipe settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit(1)

# Adjusted smoothing parameters for faster response
prev_left_right_angle = 90
prev_up_down_angle = 90
prev_claw_angle = 0
smoothing_factor = 0.2  # Increased for faster response (was 0.15)
pinch_smoothing_factor = 0.25  # Specific for pinch control

# Movement thresholds
MOVEMENT_THRESHOLD = 3
HAND_CONFIDENCE_THRESHOLD = 0.6

# Buffer for pinch distance smoothing
pinch_distance_buffer = []
BUFFER_SIZE = 5  # Reduced buffer size for quicker response

def apply_smoothing(new_value, prev_value, alpha):
    return alpha * new_value + (1 - alpha) * prev_value

def map_value(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def smooth_pinch_distance(new_distance):
    """Apply moving average smoothing to pinch distance"""
    pinch_distance_buffer.append(new_distance)
    if len(pinch_distance_buffer) > BUFFER_SIZE:
        pinch_distance_buffer.pop(0)
    return sum(pinch_distance_buffer) / len(pinch_distance_buffer)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results_pose = pose.process(img_rgb)
        results_hands = hands.process(img_rgb)

        movement_detected = False
        hand_detected = False

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            
            try:
                elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                if elbow.visibility > HAND_CONFIDENCE_THRESHOLD and wrist.visibility > HAND_CONFIDENCE_THRESHOLD:
                    movement_detected = True
                    
                    dx = wrist.x - elbow.x
                    dy = wrist.y - elbow.y

                    # Left-right movement with faster response
                    left_right_angle = np.degrees(np.arctan2(dx, 0.1)) + 90
                    left_right_angle = apply_smoothing(left_right_angle, prev_left_right_angle, smoothing_factor)
                    left_right_angle = np.clip(left_right_angle, 0, 180)

                    # Up-down movement with faster response
                    up_down_angle = map_value(wrist.y, 0.3, 0.7, 0, 180)
                    up_down_angle = apply_smoothing(up_down_angle, prev_up_down_angle, smoothing_factor)

                    # Update servos with faster response
                    if abs(left_right_angle - prev_left_right_angle) > MOVEMENT_THRESHOLD:
                        servos["left_right"].angle = left_right_angle
                        prev_left_right_angle = left_right_angle

                    if abs(up_down_angle - prev_up_down_angle) > MOVEMENT_THRESHOLD:
                        servos["up_down"].angle = up_down_angle
                        prev_up_down_angle = up_down_angle

                    # Visualization
                    h, w, _ = img.shape
                    cv2.circle(img, (int(elbow.x * w), int(elbow.y * h)), 5, (0, 255, 0), -1)
                    cv2.circle(img, (int(wrist.x * w), int(wrist.y * h)), 5, (0, 255, 0), -1)
                    cv2.line(img, 
                            (int(elbow.x * w), int(elbow.y * h)),
                            (int(wrist.x * w), int(wrist.y * h)),
                            (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing arm movement: {e}")

        # Process hands for claw control using thumb-index pinch
        if results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            hand_detected = True

            try:
                # Get thumb and index finger positions
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate pinch distance
                pinch_distance = np.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 + 
                    (thumb_tip.y - index_tip.y) ** 2
                )

                # Smooth the pinch distance
                smoothed_distance = smooth_pinch_distance(pinch_distance)

                # Control claw based on pinch distance
                claw_angle = map_value(smoothed_distance, 0.05, 0.2, 90, 0)
                claw_angle = apply_smoothing(claw_angle, prev_claw_angle, pinch_smoothing_factor)

                if abs(claw_angle - prev_claw_angle) > MOVEMENT_THRESHOLD:
                    servos["claw"].angle = claw_angle
                    prev_claw_angle = claw_angle

                # Visualize thumb and index finger
                h, w, _ = img.shape
                
                # Draw thumb point
                thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                cv2.circle(img, thumb_point, 5, (255, 0, 0), -1)  # Blue for thumb
                cv2.circle(img, thumb_point, 7, (255, 255, 255), 2)

                # Draw index point
                index_point = (int(index_tip.x * w), int(index_tip.y * h))
                cv2.circle(img, index_point, 5, (0, 0, 255), -1)  # Red for index
                cv2.circle(img, index_point, 7, (255, 255, 255), 2)

                # Draw pinch line
                cv2.line(img, thumb_point, index_point, (0, 255, 0), 2)

                # Show pinch distance
                cv2.putText(img, f"Pinch: {smoothed_distance:.2f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing hand gesture: {e}")

        # Status display
        cv2.putText(img, "Underwater Robot Arm Control", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if movement_detected else (0, 0, 255)
        cv2.putText(img, f"Arm Tracking: {'Active' if movement_detected else 'Inactive'}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        claw_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(img, f"Claw Control: {'Active' if hand_detected else 'Inactive'}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, claw_color, 2)

        cv2.imshow("Underwater Robot Arm Control", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    pca.deinit()
