import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Vision Pro | Object Detection",
    page_icon="🔍",
    layout="wide"
)

# --- PROFESSIONAL CSS (Visibility Fix) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #111111;
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
        font-size: 1.1rem;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4251;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(f"{model_name}.pt")

# --- SIDEBAR ---
st.sidebar.title("⚙️ AI Configuration")

model_size = st.sidebar.selectbox(
    "1. Select Model", 
    ["yolov8n", "yolov8s"], 
    help="Nano (n) is fastest, Small (s) is more accurate."
)

confidence = st.sidebar.slider(
    "2. Confidence Threshold", 
    0.0, 1.0, 0.45, 0.05
)

with st.sidebar.expander("🎓 Learn: How Confidence Works?"):
    st.write("The AI only shows objects it is at least certain about based on this percentage.")

st.sidebar.markdown("---")
st.sidebar.info(f"**Dev:** Piyush Sharma\n\n**Email:** sharmapiyush4845@gmail.com")

# --- MAIN INTERFACE ---
st.title("🔍 Real-Time AI Object Detection")
st.write("This dashboard uses **YOLOv8** and **WebRTC** to identify objects through your browser's webcam.")

col1, col2 = st.columns([2, 1])

# Load the model
model = load_yolo_model(model_size)

# --- WEBRTC LOGIC ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.45

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # AI Inference
        results = model(img, conf=self.confidence)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

with col1:
    st.subheader("🖥️ Live Feed")
    
    # RTC Configuration for Cloud (uses Google's STUN server)
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Update the confidence threshold in the processor
    if ctx.video_processor:
        ctx.video_processor.confidence = confidence

with col2:
    st.subheader("📊 Statistics")
    # Note: Live Metrics in WebRTC are handled inside the video stream for performance
    st.info("Performance metrics are rendered directly on the video feed.")
    
    st.markdown("### 📝 Instructions")
    st.info("""
    1. Click **START** to activate your webcam.
    2. Allow camera permissions in your browser.
    3. Use the slider on the left to adjust AI sensitivity.
    """)
