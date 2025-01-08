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
```
