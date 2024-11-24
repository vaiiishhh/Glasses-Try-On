# Glasses Overlay using OpenCV

This project allows users to add virtual glasses to their faces in real-time using a webcam. It detects faces and eyes, overlays glasses, and includes options to select different styles of glasses dynamically.

---

## Features
- Real-time face and eye detection using Haar Cascades.
- Virtual glasses overlay with smooth position tracking to reduce jitter.
- Four styles of glasses to choose from, selectable during runtime.
- Built using Python and OpenCV.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/glasses-overlay.git
   cd glasses-overlay
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy
   ```

3. **Add glasses images**:
   - Place your `.png` glasses images (with transparency) in the project directory.
   - Ensure filenames are `sunglasses1.png`, `sunglasses2.png`, `specs1.png`, and `specs2.png` for seamless usage.

---

## Usage

1. **Run the program**:
   ```bash
   python main.py
   ```

2. **Controls**:
   - Press `1`, `2`, `3`, or `4` to switch between different glasses styles.
   - Press `q` to quit the application.

---

## Files
- **`main.py`**: Entry point; manages webcam feed and user interaction.
- **`glasses_overlay.py`**: Contains functions for detecting faces/eyes and overlaying glasses.
- **`haarcascades`**: Pre-trained Haar Cascade models for face and eye detection.

---
