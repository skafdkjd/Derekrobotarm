import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands with different configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,  # Use simpler model
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        break
    
    # Flip the image horizontally for a later selfie-view display
    img = cv2.flip(img, 1)
    
    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a black background
    black_bg = np.zeros_like(img)
    
    # Process the image
    results = hands.process(img_rgb)
    
    # If hands are detected, draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_draw.draw_landmarks(
                black_bg,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Add labels for fingertips
            h, w, c = img.shape
            finger_tips = {
                4: "T",   # Thumb
                8: "I",   # Index
                12: "M",  # Middle
                16: "R",  # Ring
                20: "P"   # Pinky
            }
            
            for tip_id, finger_name in finger_tips.items():
                x = int(hand_landmarks.landmark[tip_id].x * w)
                y = int(hand_landmarks.landmark[tip_id].y * h)
                cv2.putText(black_bg, finger_name, (x-10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Overlay the black background with the original image
    overlay = cv2.addWeighted(img, 0.5, black_bg, 0.5, 0)
    
    # Display instructions
    cv2.putText(overlay, "Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Display the image
    cv2.imshow("Hand Tracking", overlay)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()  # Ensure to close the hands object to release resources