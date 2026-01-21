import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision Pro", page_icon="🔍", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #111111; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4251; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(f"{model_name}.pt")

# --- SIDEBAR ---
st.sidebar.title("⚙️ AI Configuration")
model_size = st.sidebar.selectbox("1. Select Model", ["yolov8n", "yolov8s"])
confidence = st.sidebar.slider("2. Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

# TAB SELECTION
app_mode = st.sidebar.radio("3. Choose Input Mode", ["Image Upload (Cloud Ready)", "Webcam (Local Only)"])

with st.sidebar.expander("🎓 Learn: How Confidence Works?"):
    st.write("AI calculates a probability score. The threshold filters out weak guesses.")

st.sidebar.info(f"**Dev:** Piyush Sharma")

# --- MAIN INTERFACE ---
st.title("🔍 Real-Time AI Object Detection")
model = load_yolo_model(model_size)

col1, col2 = st.columns([2, 1])

# --- MODE: IMAGE UPLOAD ---
if app_mode == "Image Upload (Cloud Ready)":
    with col1:
        st.subheader("🖼️ Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = np.array(Image.open(uploaded_file))
            results = model(image, conf=confidence)
            annotated_img = results[0].plot()
            st.image(annotated_img, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("📊 Statistics")
                st.metric("Objects Detected", len(results[0].boxes))
                st.success("Analysis Complete!")

# --- MODE: WEBCAM ---
elif app_mode == "Webcam (Local Only)":
    with col1:
        st.subheader("🖥️ Live Feed")
        run_webcam = st.toggle("🚀 Start Live Detection")
        frame_placeholder = st.empty()
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            while cap.isOpened() and run_webcam:
                success, frame = cap.read()
                if not success: break
                
                results = model(frame, conf=confidence, stream=True)
                for r in results:
                    annotated_frame = r.plot()
                    count = len(r.boxes)
                
                frame_placeholder.image(annotated_frame, channels="BGR")
                with col2:
                    st.subheader("📊 Statistics")
                    st.metric("Objects in View", count)
            cap.release()
        else:
            idle_img = np.zeros((480, 640, 3), dtype=np.uint8) + 40
            cv2.putText(idle_img, "Webcam Mode: Use locally for best results.", (80, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            frame_placeholder.image(idle_img, channels="RGB")
