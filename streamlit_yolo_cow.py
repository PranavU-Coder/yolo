import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

st.set_page_config(page_title="Different YOLO Models", layout="centered")
st.title("YOLO Models")

st.divider()

# am removing all the gdown commands since the final container will have all the models in it locally to run

MODEL_OPTIONS = {
    "YOLOv8 Nano": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
    "YOLOv8 Extra-Large": "yolov8x.pt"
}

model_name = st.sidebar.selectbox("Choose a YOLO model", list(MODEL_OPTIONS.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Load pre-installed model

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(MODEL_OPTIONS[model_name])

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    is_video = uploaded_file.type.startswith("video")

    if not is_video:
        image = Image.open(file_path).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Running detection..."):
            results = model(image, conf=confidence_threshold)
            result_img = results[0].plot()
        
        st.image(result_img, caption="Detected", use_container_width=True)
        
        # Show detection stats
        detections = results[0].boxes
        st.info(f"Detected {len(detections)} objects")
        
    else:
        st.video(file_path)
        cap = cv2.VideoCapture(file_path)

        st.info("Processing video (showing 10 annotated frames)...")
        frame_count = 0
        
        while cap.isOpened() and frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, conf=confidence_threshold)
            annotated = results[0].plot()
            st.image(annotated, caption=f"Frame {frame_count + 1}", use_container_width=True)
            frame_count += 1
            
        cap.release()
        st.success(f"Processed {frame_count} frames")