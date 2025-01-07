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

# Create servo objects for each finger
servos = {
    "thumb": servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400),
    "index": servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400),
    "middle": servo.Servo(pca.channels[2], min_pulse=600, max_pulse=2400),
    "ring": servo.Servo(pca.channels[3], min_pulse=600, max_pulse=2400),
    "pinky": servo.Servo(pca.channels[4], min_pulse=600, max_pulse=2400)
}

# Initialize MediaPipe Hands
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

def calculate_finger_angle(landmark1, landmark2, landmark3):
    """Calculate angle between three points"""
    v1 = np.array([landmark1.x, landmark1.y])
    v2 = np.array([landmark2.x, landmark2.y])
    v3 = np.array([landmark3.x, landmark3.y])
    
    # Calculate vectors
    vector1 = v1 - v2
    vector2 = v3 - v2
    
    # Calculate angle
    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def map_angle_to_servo(angle, in_min=0, in_max=180, out_min=0, out_max=180):
    """Map angle to servo range"""
    return int((angle - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        black_bg = np.zeros_like(img)
        
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            
            # Draw landmarks
            mp_draw.draw_landmarks(
                black_bg,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Process each finger
            finger_angles = {
                "thumb": calculate_finger_angle(
                    hand_landmarks.landmark[4],  # Thumb tip
                    hand_landmarks.landmark[3],  # Thumb IP
                    hand_landmarks.landmark[2]   # Thumb MCP
                ),
                "index": calculate_finger_angle(
                    hand_landmarks.landmark[8],  # Index tip
                    hand_landmarks.landmark[6],  # Index PIP
                    hand_landmarks.landmark[5]   # Index MCP
                ),
                "middle": calculate_finger_angle(
                    hand_landmarks.landmark[12], 
                    hand_landmarks.landmark[10],
                    hand_landmarks.landmark[9]
                ),
                "ring": calculate_finger_angle(
                    hand_landmarks.landmark[16],
                    hand_landmarks.landmark[14],
                    hand_landmarks.landmark[13]
                ),
                "pinky": calculate_finger_angle(
                    hand_landmarks.landmark[20],
                    hand_landmarks.landmark[18],
                    hand_landmarks.landmark[17]
                )
            }
            
            # Control servos based on finger angles
            for finger, angle in finger_angles.items():
                servo_angle = map_angle_to_servo(angle)
                servos[finger].angle = servo_angle
                
                # Display servo angles on screen
                h, w, c = img.shape
                landmark_idx = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
                x = int(hand_landmarks.landmark[landmark_idx[finger]].x * w)
                y = int(hand_landmarks.landmark[landmark_idx[finger]].y * h)
                cv2.putText(black_bg, f"{finger}: {int(servo_angle)}°", (x-30, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        overlay = cv2.addWeighted(img, 0.5, black_bg, 0.5, 0)
        cv2.putText(overlay, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow("Hand Tracking with Servo Control", overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pca.deinit()
