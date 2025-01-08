# Robotic Arm Control with MediaPipe and PCA9685

This project uses MediaPipe Pose and Hands to control a robotic arm with servos. The robotic arm mimics the movements of a human arm and detects hand gestures to control an additional servo for grabbing actions.

## Features

- **Arm Tracking:** Uses MediaPipe Pose to track the human arm and map the movements to the robotic arm's servos.
- **Hand Gesture Detection:** Uses MediaPipe Hands to detect hand gestures (open/close) and control a grab servo.
- **Servo Control:** Controls up to 6 servos using the PCA9685 PWM driver.

## Hardware Requirements

- Raspberry Pi (or compatible device)
- PCA9685 PWM driver
- Servos (up to 6)
- Webcam

## Software Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- Adafruit PCA9685 library
- Adafruit Motor library

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/keyframesfound/robotic-arm.git
   cd robotic-arm
   ```

2. **Install the required Python libraries:**
   ```sh
   pip install opencv-python mediapipe adafruit-circuitpython-pca9685 adafruit-circuitpython-servokit
   ```

## Usage

1. **Connect the hardware:**
   - Connect the servos to the PCA9685 PWM driver.
   - Connect the PCA9685 to the Raspberry Pi using I2C.
   - Connect the webcam to the Raspberry Pi.

2. **Run the script:**
   ```sh
   python testing.py
   ```

## License

This project is licensed under the MIT License. See the 

LICENSE

 file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for providing the Pose and Hands solutions.
- [Adafruit](https://www.adafruit.com/) for the PCA9685 and servo libraries.


## Technical Documentation

### Code Overview

The code controls a robotic arm using MediaPipe Pose and Hands to track human arm movements and hand gestures. The robotic arm is controlled via servos connected to a PCA9685 PWM driver.

### Initialization

The initialization sets up the necessary libraries, initializes the I2C bus and PCA9685, and creates servo objects for each joint and an additional servo for the grab action.

```python
import mediapipe as mp
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import board
import busio
import time
from collections import deque

# Initialize I2C bus and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Set PWM frequency to 50Hz (standard for servos)

# Create servo objects for each joint and an additional servo for the grab action
servos = {
    "base": servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400),  # Base rotation
    "shoulder": servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400),
    "elbow": servo.Servo(pca.channels[2], min_pulse=600, max_pulse=2400),
    "wrist": servo.Servo(pca.channels[3], min_pulse=600, max_pulse=2400),
    "wrist_rotate": servo.Servo(pca.channels[4], min_pulse=600, max_pulse=2400),  # Wrist rotation
    "grab": servo.Servo(pca.channels[5], min_pulse=600, max_pulse=2400)  # Additional servo for grab action
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
```

### Main Loop

The main loop captures frames from the webcam, processes them using MediaPipe Pose and Hands, calculates angles for the arm joints, and controls the servos accordingly. It also detects hand gestures to control the grab servo.

```python
# Deques for smoothing angles
base_angles = deque(maxlen=5)
shoulder_angles = deque(maxlen=5)
elbow_angles = deque(maxlen=5)
wrist_angles = deque(maxlen=5)
wrist_rotate_angles = deque(maxlen=5)

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

def smooth_angle(angle, angle_deque):
    """Smooth the angle using a moving average"""
    angle_deque.append(angle)
    return np.mean(angle_deque)

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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Calculate angles
            base_angle = calculate_angle(hip, shoulder, elbow)
            shoulder_angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angle = calculate_angle(elbow, shoulder, wrist)
            wrist_angle = calculate_angle(wrist, elbow, shoulder)
            wrist_rotate_angle = calculate_angle(elbow, wrist, [wrist[0], wrist[1] + 0.1])  # Simplified wrist rotation
            
            # Smooth angles
            base_angle = smooth_angle(base_angle, base_angles)
            shoulder_angle = smooth_angle(shoulder_angle, shoulder_angles)
            elbow_angle = smooth_angle(elbow_angle, elbow_angles)
            wrist_angle = smooth_angle(wrist_angle, wrist_angles)
            wrist_rotate_angle = smooth_angle(wrist_rotate_angle, wrist_rotate_angles)
            
            # Map angles to servo range
            base_servo_angle = map_angle_to_servo(base_angle)
            shoulder_servo_angle = map_angle_to_servo(shoulder_angle)
            elbow_servo_angle = map_angle_to_servo(elbow_angle)
            wrist_servo_angle = map_angle_to_servo(wrist_angle)
            wrist_rotate_servo_angle = map_angle_to_servo(wrist_rotate_angle)
            
            # Control servos
            servos["base"].angle = base_servo_angle
            servos["shoulder"].angle = shoulder_servo_angle
            servos["elbow"].angle = elbow_servo_angle
            servos["wrist"].angle = wrist_servo_angle
            servos["wrist_rotate"].angle = wrist_rotate_servo_angle
            
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
            
            if distance < 0.1:  # Increased threshold for detecting a grab
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
```

### Functions

- **calculate_angle(a, b, c):** Calculates the angle between three points.
- **map_angle_to_servo(angle, in_min, in_max, out_min, out_max):** Maps an angle to the servo range.
- **calculate_distance(point1, point2):** Calculates the Euclidean distance between two points.
- **smooth_angle(angle, angle_deque):** Smooths the angle using a moving average.

### Servo Control

The servos are controlled based on the calculated angles from the pose landmarks. The angles are smoothed using a moving average to reduce jitter and then mapped to the servo range.

### Hand Gesture Detection

The distance between the thumb tip and index tip is used to detect a grab action. If the distance is below a certain threshold, the grab servo is activated.

### Cleanup

The `finally` block ensures that the resources are properly released when the script is terminated.

```python
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    pca.deinit()
```