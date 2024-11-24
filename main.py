import cv2
import numpy as np
import os
from glasses_overlay import overlay_glasses

def load_glasses_images(cd):
    glasses_files = ['sunglasses1.png', 'sunglasses2.png', 'specs1.png', 'specs2.png']
    glasses_images = [cv2.imread(os.path.join(cd, file), cv2.IMREAD_UNCHANGED) for file in glasses_files]
    return [g for g in glasses_images if g is not None]

def main():
    cd = os.path.dirname(os.path.abspath(__file__))
    glasses_images = load_glasses_images(cd)

    if not glasses_images:
        print("No glasses images found!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    selected_glasses = glasses_images[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        result = overlay_glasses(frame, selected_glasses)

        instructions = "Press 1, 2, 3, or 4 to select glasses. Press 'q' to quit."
        cv2.putText(result, instructions, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Glasses Overlay", result)

        key = cv2.waitKey(1)
        if key == ord('1'):
            selected_glasses = glasses_images[0]
        elif key == ord('2'):
            selected_glasses = glasses_images[1]
        elif key == ord('3'):
            selected_glasses = glasses_images[2]
        elif key == ord('4'):
            selected_glasses = glasses_images[3]
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()