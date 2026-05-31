import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# COMPLETE HOME OBJECTS + YOUR REQUESTS
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

st.title("Live Detection")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

frame_placeholder = st.empty()
objects_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, classes=HOME_CLASSES, persist=True, verbose=False)
    annotated_frame = results[0].plot()
    
    frame_placeholder.image(annotated_frame, channels="BGR", width=800)
    
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf > 0.4:
                detections.append(f"#{cls_id} ({conf:.1f})")
    
    objects_placeholder.markdown("**Live:** " + ", ".join(detections) if detections else "Scanning...")

cap.release()