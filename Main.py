import cv2
import numpy as np
import os

# Load pre-trained cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load sunglasses image
script_dir = os.path.dirname(os.path.abspath(__file__))
sunglasses_path = os.path.join(script_dir, 'sunglasses.png')
sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)

# Smoothing variables
previous_positions = {}
smoothing_factor = 1

def smooth_position(face_id, current_pos):
    if face_id not in previous_positions:
        previous_positions[face_id] = current_pos
        return current_pos
    
    # Interpolate between previous and current position
    prev_pos = previous_positions[face_id]
    smoothed_pos = (
        int(current_pos[0] * smoothing_factor + prev_pos[0] * (1 - smoothing_factor)),
        int(current_pos[1] * smoothing_factor + prev_pos[1] * (1 - smoothing_factor))
    )
    previous_positions[face_id] = smoothed_pos
    return smoothed_pos

def overlay_sunglasses(frame, sunglasses):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Clean up previous positions for faces no longer in frame
    for face_id in list(previous_positions.keys()):
        if not any(np.array_equal(face[:2], (face_id[0], face_id[1])) for face in faces):
            del previous_positions[face_id]
    
    for (x, y, w, h) in faces:
        # Detect eyes within the face region
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Sort eyes from left to right
            eyes = sorted(eyes, key=lambda eye: eye[0])
            
            # Calculate sunglasses sizing and positioning
            eye_distance = abs(eyes[1][0] - eyes[0][0]) + eyes[0][2] + eyes[1][2]
            
            # Sizing and positioning
            size_multiplier = 1.5
            vertical_offset = 0.15
            
            # Resize sunglasses
            resized_width = int(eye_distance * size_multiplier)
            resized_height = int(sunglasses.shape[0] * resized_width / sunglasses.shape[1])
            resized_sunglasses = cv2.resize(sunglasses, (resized_width, resized_height))
            
            # Calculate positioning
            eye_center_x = x + (eyes[0][0] + eyes[1][0] + eyes[0][2]//2 + eyes[1][2]//2) // 2
            eye_center_y = y + int((eyes[0][1] + eyes[1][1]) // 2 + h * vertical_offset)
            
            # Smooth the positioning using a unique face identifier
            current_pos = (
                eye_center_x - resized_sunglasses.shape[1] // 2,
                eye_center_y - resized_sunglasses.shape[0] // 2
            )
            face_id = (x, y)  # Unique identifier for each face
            smoothed_pos = smooth_position(face_id, current_pos)
            
            overlay_image(frame, resized_sunglasses, smoothed_pos[0], smoothed_pos[1])
    
    return frame

def overlay_image(background, overlay, x_offset, y_offset):
    # Boundary check
    if (x_offset < 0 or y_offset < 0 or 
        x_offset + overlay.shape[1] > background.shape[1] or 
        y_offset + overlay.shape[0] > background.shape[0]):
        return
    
    # Check if overlay has alpha channel
    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        
        for c in range(0, 3):
            background[y_offset:y_offset+overlay.shape[0], 
                      x_offset:x_offset+overlay.shape[1], c] = (
                alpha_overlay * overlay[:, :, c] +
                alpha_background * background[y_offset:y_offset+overlay.shape[0], 
                                              x_offset:x_offset+overlay.shape[1], c]
            )
    else:
        background[y_offset:y_offset+overlay.shape[0], 
                   x_offset:x_offset+overlay.shape[1]] = overlay

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect faces and overlay sunglasses
    result = overlay_sunglasses(frame, sunglasses)
    
    cv2.imshow('Sunglasses Overlay', result)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()