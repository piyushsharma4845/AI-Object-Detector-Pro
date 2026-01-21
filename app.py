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

# --- PROFESSIONAL CSS (Kept Exactly Same) ---
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

# --- SIDEBAR: SETTINGS ---
st.sidebar.title("⚙️ AI Configuration")

model_size = st.sidebar.selectbox(
    "1. Select Model", 
    ["yolov8n", "yolov8s"]
)

confidence = st.sidebar.slider(
    "2. Confidence Threshold", 
    0.0, 1.0, 0.45, 0.05
)

with st.sidebar.expander("🎓 Learn: How Confidence Works?"):
    st.write("Filters out detections below this probability score.")

st.sidebar.markdown("---")
# YOUR TOGGLE BUTTON
run_webcam = st.sidebar.toggle("🚀 Start Live Detection")

st.sidebar.markdown("---")
st.sidebar.info(f"**Dev:** Piyush Sharma\n\n**Email:** sharmapiyush4845@gmail.com")

# --- MAIN INTERFACE ---
st.title("🔍 Real-Time AI Object Detection")
st.write("This dashboard uses **YOLOv8** to identify objects in real-time.")

col1, col2 = st.columns([2, 1])

# Load Model
model = load_yolo_model(model_size)

# --- AI PROCESSING CLASS ---
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.45

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=self.confidence)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

with col1:
    st.subheader("🖥️ Live Feed")
    
    if run_webcam:
        # This replaces cv2.VideoCapture to work on the internet
        ctx = webrtc_streamer(
            key="yolo",
            video_processor_factory=YOLOProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.confidence = confidence
    else:
        # Your Idle Grey Box
        idle_img = np.zeros((480, 640, 3), dtype=np.uint8) + 40 
        cv2.putText(idle_img, "Webcam is Idle. Flip toggle to start.", (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        st.image(idle_img, channels="RGB")

with col2:
    st.subheader("📊 Statistics")
    st.info("Metrics are displayed directly on the video feed for real-time performance.")
    
    st.markdown("### 📝 Instructions")
    st.info("""
    1. Select model from sidebar.
    2. Toggle **Start Live Detection**.
    3. Click the **'START'** button that appears in the feed.
    4. Allow browser camera permissions.
    """)
