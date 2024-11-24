import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

previous_positions = {}
smoothing_factor = 0.8

def smooth_position(face_id, current_pos):
    if face_id not in previous_positions:
        previous_positions[face_id] = current_pos
        return current_pos
    prev_pos = previous_positions[face_id]
    smoothed_pos = (
        int(prev_pos[0] * smoothing_factor + current_pos[0] * (1 - smoothing_factor)),
        int(prev_pos[1] * smoothing_factor + current_pos[1] * (1 - smoothing_factor))
    )
    previous_positions[face_id] = smoothed_pos
    return smoothed_pos

def overlay_image(background, overlay, x_offset, y_offset):
    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        for c in range(3):
            background[y_offset:y_offset + overlay.shape[0], 
                       x_offset:x_offset + overlay.shape[1], c] = (
                alpha_overlay * overlay[:, :, c] +
                alpha_background * background[y_offset:y_offset + overlay.shape[0], 
                                              x_offset:x_offset + overlay.shape[1], c]
            )

def overlay_glasses(frame, glasses):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    
    for face_id in list(previous_positions.keys()):
        if not any((x, y) == face_id for x, y, w, h in faces):
            del previous_positions[face_id]
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            eye_distance = abs(eyes[1][0] - eyes[0][0]) + eyes[0][2] + eyes[1][2]
            
            resized_width = int(eye_distance * 1.5)
            resized_height = int(glasses.shape[0] * resized_width / glasses.shape[1])
            resized_glasses = cv2.resize(glasses, (resized_width, resized_height))
            
            eye_center_x = x + (eyes[0][0] + eyes[1][0] + eyes[0][2]//2 + eyes[1][2]//2) // 2
            eye_center_y = y + int((eyes[0][1] + eyes[1][1]) // 2 + h * 0.15)
            
            current_pos = (
                eye_center_x - resized_glasses.shape[1] // 2,
                eye_center_y - resized_glasses.shape[0] // 2
            )
            face_id = (x, y)
            smoothed_pos = smooth_position(face_id, current_pos)
            
            overlay_image(frame, resized_glasses, smoothed_pos[0], smoothed_pos[1])
    return frame
