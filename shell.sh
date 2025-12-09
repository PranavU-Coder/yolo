#!/bin/bash

pip install ultralytics

# downloading all the yolo models

python3 << EOF
from ultralytics import YOLO

models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']

for model_name in models:
    YOLO(model_name)
    
print("all models downloaded successfully")
EOF

# running the streamlit website now in the end

streamlit run streamlit_yolo_cow.py --server.port=8501 --server.address=0.0.0.0