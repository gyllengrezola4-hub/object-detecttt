 Home Object Detection

A real-time computer vision application that detects common household objects using your webcam. Built with [Streamlit](https://streamlit.io/), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), and OpenCV.

## Features

- **Real-Time Detection**: Live video feed from your webcam
- **Smart Filtering**: Only detects relevant home objects (laptops, cups, phones, etc.)
- **Object Tracking**: Maintains IDs across frames for moving objects
- **Confidence Threshold**: Filters weak detections (>40% confidence)

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install streamlit ultralytics opencv-python numpy
Run the App
bash

Copy code
streamlit run app.py
Supported Objects
ID

Object

0

Person

23

Handbag

24

Backpack

40

Cup

41

Fork

42

Spoon

43

Bowl

56

Chair

62

Laptop

67

Cellphone

Tech Stack
Frontend: Streamlit
Model: YOLOv8 Nano (COCO weights)
Backend: Python, OpenCV
License
MIT


Copy code

---

## 2. In-Code Documentation (Python Comments)

Replace your `app.py` with this fully commented version:

```python
"""
Home Object Detection App
========================
A Streamlit web app that performs real-time object detection on webcam feed
using YOLOv8. Detects specific household items and displays them live.
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Set Streamlit page layout to wide mode for better visibility of video feed
st.set_page_config(layout="wide")

# -----------------------------------------------------------------------------
# Model Loading (Cached)
# -----------------------------------------------------------------------------
# We use @st.cache_resource to load the model only ONCE.
# Without this, Streamlit would reload the model on every interaction,
# causing massive memory usage and lag.
@st.cache_resource
def load_model():
    # Loads YOLOv8 Nano (smallest/fastest model)
    # Automatically downloads 'yolov8n.pt' if not present
    return YOLO("yolov8n.pt")

# Initialize the model
model = load_model()

# -----------------------------------------------------------------------------
# Class Definitions
# -----------------------------------------------------------------------------
# YOLO uses COCO dataset class IDs. We define a whitelist of IDs
# relevant to a home environment to filter out distractions (cars, trees, etc.)
#
# COCO Class IDs:
# 0: person, 23: handbag, 24: backpack, 40: cup, 41: fork,
# 42: spoon, 43: bowl, 56: chair, 62: laptop, 67: cellphone
HOME_CLASSES = [
    0,   # person
    23,  # handbag (sunglasses case)
    24,  # backpack (bags)
    40,  # cup
    41,  # fork 
    42,  # spoon 
    43,  # bowl
    56,  # chair
    62,  # laptop
    67,  # cellphone 
]

# -----------------------------------------------------------------------------
# UI Setup
# -----------------------------------------------------------------------------

st.title("Live Detection")

# Initialize Webcam
# args: 0 = default camera, 800x600 = resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Placeholders for dynamic updating
# These containers will be updated every frame without refreshing the whole page
frame_placeholder = st.empty()
objects_placeholder = st.empty()

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

while cap.isOpened():
    # 1. Read frame from webcam
    ret, frame = cap.read()
    
    # Exit loop if camera disconnected or error
    if not ret:
        break
    
    # 2. Run Detection
    # model.track() is used instead of predict() to enable object tracking
    # persist=True keeps tracking IDs stable across frames
    # verbose=False suppresses terminal output logs
    # classes=HOME_CLASSES filters to only our allowed objects
    results = model.track(frame, classes=HOME_CLASSES, persist=True, verbose=False)
    
    # 3. Annotate Frame
    # results[0].plot() draws bounding boxes and labels onto the image
    annotated_frame = results[0].plot()
    
    # 4. Update UI
    # Convert BGR (OpenCV format) to RGB for Streamlit display
    frame_placeholder.image(annotated_frame, channels="BGR", width=800)
    
    # 5. Extract Detection Data for List Display
    detections = []
    
    # Check if any boxes were detected
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Get class ID and confidence score
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            # Filter: Only show high-confidence detections (>40%)
            if conf > 0.4:
                # Format: #ClassID (Confidence)
                detections.append(f"#{cls_id} ({conf:.1f})")
    
    # Display list of detected objects
    # Use markdown for bold text styling
    objects_placeholder.markdown("**Live:** " + ", ".join(detections) if detections else "Scanning...")

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
# Release camera resources when loop ends
cap.release()
