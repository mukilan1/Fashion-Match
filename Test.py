import cv2
import mediapipe as mp
import numpy as np
import os

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define custom colors for different body parts (in BGR format)
colors = {
    'right_arm': (255, 0, 0),      # Blue
    'left_arm': (0, 255, 0),       # Green
    'right_leg': (0, 0, 255),      # Red
    'left_leg': (255, 255, 0),     # Cyan
    'torso': (255, 0, 255),        # Magenta
    'face': (0, 255, 255)          # Yellow
}

# Define body part connections
body_parts = {
    'right_arm': [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                 (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)],
    'left_arm': [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)],
    'right_leg': [(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                 (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)],
    'left_leg': [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)],
    'torso': [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
             (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
             (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
             (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER)],
    'face': [(mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
            (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
            (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
            (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
            (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
            (mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT)]
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = pose.process(rgb_frame)
    
    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Draw custom colored body parts with thicker lines
        for part, connections in body_parts.items():
            color = colors[part]
            for connection in connections:
                start_idx, end_idx = connection
                if (landmarks[start_idx].visibility > 0.5 and 
                    landmarks[end_idx].visibility > 0.5):
                    
                    start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    
                    # Draw thicker lines (adjust thickness as needed)
                    cv2.line(frame, start_point, end_point, color, 6)
        
        # Draw larger circles at key joints
        key_points = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                     mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                     mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                     mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                     mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                     mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
                     mp_pose.PoseLandmark.NOSE]
        
        for point in key_points:
            if landmarks[point].visibility > 0.5:
                point_x = int(landmarks[point].x * w)
                point_y = int(landmarks[point].y * h)
                # Draw a larger filled circle at each key point
                cv2.circle(frame, (point_x, point_y), 8, (255, 255, 255), -1)  # White filled circle
                cv2.circle(frame, (point_x, point_y), 8, (0, 0, 0), 2)         # Black outline
                
    # Display the resulting frame
    cv2.imshow("Enhanced MediaPipe Pose Rigging", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()