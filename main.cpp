#include <opencv2/opencv.hpp>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <wiringPiI2C.h>
#include <cmath>
#include <deque>
#include <iostream>

// Constants
const int PCA9685_ADDR = 0x40;
const int PCA9685_MODE1 = 0x00;
const int PCA9685_PRESCALE = 0xFE;
const int SERVO_MIN = 150;  // Min pulse length out of 4096
const int SERVO_MAX = 600;  // Max pulse length out of 4096

// Function to set PWM frequency
void setPWMFreq(int fd, float freq) {
    float prescaleval = 25000000.0; // 25MHz
    prescaleval /= 4096.0;          // 12-bit
    prescaleval /= freq;
    prescaleval -= 1.0;
    int prescale = floor(prescaleval + 0.5);
    int oldmode = wiringPiI2CReadReg8(fd, PCA9685_MODE1);
    int newmode = (oldmode & 0x7F) | 0x10; // sleep
    wiringPiI2CWriteReg8(fd, PCA9685_MODE1, newmode); // go to sleep
    wiringPiI2CWriteReg8(fd, PCA9685_PRESCALE, prescale); // set the prescaler
    wiringPiI2CWriteReg8(fd, PCA9685_MODE1, oldmode);
    delay(5);
    wiringPiI2CWriteReg8(fd, PCA9685_MODE1, oldmode | 0x80);
}

// Function to set PWM
void setPWM(int fd, int channel, int on, int off) {
    wiringPiI2CWriteReg8(fd, 0x06 + 4 * channel, on & 0xFF);
    wiringPiI2CWriteReg8(fd, 0x07 + 4 * channel, on >> 8);
    wiringPiI2CWriteReg8(fd, 0x08 + 4 * channel, off & 0xFF);
    wiringPiI2CWriteReg8(fd, 0x09 + 4 * channel, off >> 8);
}

// Function to map angle to servo pulse
int mapAngleToServo(float angle, float in_min = 0, float in_max = 180, int out_min = SERVO_MIN, int out_max = SERVO_MAX) {
    return (int)((angle - in_min) * (out_max - out_min) / (in_max - in_min) + out_min);
}

// Function to calculate angle between three points
float calculateAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
    float radians = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x);
    float angle = std::abs(radians * 180.0 / CV_PI);
    if (angle > 180.0) {
        angle = 360.0 - angle;
    }
    return angle;
}

// Function to calculate Euclidean distance between two points
float calculateDistance(cv::Point2f point1, cv::Point2f point2) {
    return cv::norm(point1 - point2);
}

int main() {
    // Initialize I2C and PCA9685
    int fd = wiringPiI2CSetup(PCA9685_ADDR);
    setPWMFreq(fd, 50); // Set frequency to 50Hz

    // Initialize MediaPipe Pose and Hands
    // (Initialization code for MediaPipe Pose and Hands goes here)

    // Initialize webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // Deques for smoothing angles
    std::deque<float> shoulder_angles, elbow_angles, wrist_angles;

    while (true) {
        cv::Mat img;
        cap >> img;
        if (img.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        cv::flip(img, img, 1);
        cv::Mat img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

        // Process pose and hands
        // (Processing code for MediaPipe Pose and Hands goes here)

        // Example landmarks (replace with actual landmarks from MediaPipe)
        cv::Point2f shoulder(0.5, 0.5), elbow(0.6, 0.6), wrist(0.7, 0.7);
        cv::Point2f thumb_tip(0.8, 0.8), index_tip(0.9, 0.9);

        // Calculate angles
        float shoulder_angle = calculateAngle(shoulder, elbow, wrist);
        float elbow_angle = calculateAngle(elbow, shoulder, wrist);
        float wrist_angle = calculateAngle(wrist, elbow, shoulder);

        // Smooth angles
        shoulder_angles.push_back(shoulder_angle);
        elbow_angles.push_back(elbow_angle);
        wrist_angles.push_back(wrist_angle);
        if (shoulder_angles.size() > 5) shoulder_angles.pop_front();
        if (elbow_angles.size() > 5) elbow_angles.pop_front();
        if (wrist_angles.size() > 5) wrist_angles.pop_front();
        shoulder_angle = std::accumulate(shoulder_angles.begin(), shoulder_angles.end(), 0.0) / shoulder_angles.size();
        elbow_angle = std::accumulate(elbow_angles.begin(), elbow_angles.end(), 0.0) / elbow_angles.size();
        wrist_angle = std::accumulate(wrist_angles.begin(), wrist_angles.end(), 0.0) / wrist_angles.size();

        // Map angles to servo range
        int shoulder_servo_angle = mapAngleToServo(shoulder_angle);
        int elbow_servo_angle = mapAngleToServo(elbow_angle);
        int wrist_servo_angle = mapAngleToServo(wrist_angle);

        // Control servos
        setPWM(fd, 0, 0, shoulder_servo_angle);
        setPWM(fd, 1, 0, elbow_servo_angle);
        setPWM(fd, 2, 0, wrist_servo_angle);

        // Check for grab action
        float distance = calculateDistance(thumb_tip, index_tip);
        if (distance < 0.05) {  // Threshold for detecting a grab
            setPWM(fd, 3, 0, mapAngleToServo(90));  // Move the grab servo to 90 degrees
        } else {
            setPWM(fd, 3, 0, mapAngleToServo(0));  // Move the grab servo back to 0 degrees
        }

        // Display the image
        cv::imshow("Arm and Hand Tracking with Servo Control", img);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    return 0;
}