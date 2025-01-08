import cv2
import mediapipe as mp
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import board
import busio
import time

# Initialize I2C bus and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set PWM frequency to 50Hz (standard for servos)

# Create servo objects for each joint and an additional servo for the grab action
servos = {
    "shoulder": servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400),
    "elbow": servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400),
    "wrist": servo.Servo(pca.channels[2], min_pulse=600, max_pulse=2400),
    "grab": servo.Servo(pca.channels[3], min_pulse=600, max_pulse=2400)  # Additional servo for grab action
}

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,  # Changed to 1 hand for simplicity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)


def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def map_angle_to_servo(angle, in_min=0, in_max=180, out_min=0, out_max=180):
    """Map angle to servo range"""
    return int((angle - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))


try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(img_rgb)
        results_hands = hands.process(img_rgb)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angles
            shoulder_angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angle = calculate_angle(elbow, shoulder, wrist)
            wrist_angle = calculate_angle(wrist, elbow, shoulder)

            # Map angles to servo range
            shoulder_servo_angle = map_angle_to_servo(shoulder_angle)
            elbow_servo_angle = map_angle_to_servo(elbow_angle)
            wrist_servo_angle = map_angle_to_servo(wrist_angle)

            # Control servos
            servos["shoulder"].angle = shoulder_servo_angle
            servos["elbow"].angle = elbow_servo_angle
            servos["wrist"].angle = wrist_servo_angle

            # Draw landmarks and connections
            mp_draw.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0]  # Get first hand

            # Draw landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Check for grab action
            thumb_tip = [hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y]
            index_tip = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]
            distance = calculate_distance(thumb_tip, index_tip)

            if distance < 0.05:  # Threshold for detecting a grab
                servos["grab"].angle = 90  # Move the grab servo to 90 degrees
            else:
                servos["grab"].angle = 0  # Move the grab servo back to 0 degrees

        cv2.putText(img, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Arm and Hand Tracking with Servo Control", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    pca.deinit()