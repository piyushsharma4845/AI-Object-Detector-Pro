import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Vision Pro | Object Detection",
    page_icon="🔍",
    layout="wide"
)

# --- PROFESSIONAL CSS (Visibility Fix) ---
st.markdown("""
    <style>
    /* Force sidebar text to be readable in all modes */
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

# --- MODEL LOADING (Optimized) ---
@st.cache_resource
def load_yolo_model(model_name):
    return YOLO(f"{model_name}.pt")

# --- SIDEBAR: SETTINGS & EDUCATION ---
st.sidebar.title("⚙️ AI Configuration")

model_size = st.sidebar.selectbox(
    "1. Select Model", 
    ["yolov8n", "yolov8s"], 
    help="Nano (n) is fastest for webcams. Small (s) is more accurate but heavier."
)

# Confidence Slider
confidence = st.sidebar.slider(
    "2. Confidence Threshold", 
    0.0, 1.0, 0.45, 0.05,
    help="Adjust how 'sure' the AI must be to show an object."
)

# --- NEW EDUCATIONAL SECTION ---
with st.sidebar.expander("🎓 Learn: How Confidence Works?"):
    st.write("""
    **What is this slider?**
    In AI, the model doesn't just "see" a bottle; it calculates a **Probability Score** (0 to 1).
    
    * **Setting to 0.45:** The AI will only draw a box if it is at least **45% sure** the object is correct.
    * **Lowering it (0.10):** The AI guesses more. You'll see more boxes, but many will be wrong (False Positives).
    * **Raising it (0.90):** The AI becomes "strict." It only shows objects it is almost 100% certain about.
    """)

st.sidebar.markdown("---")
run_webcam = st.sidebar.toggle("🚀 Start Live Detection")

st.sidebar.markdown("---")
st.sidebar.info(f"**Dev:** Piyush Sharma\n\n**Email:** sharmapiyush4845@gmail.com")

# --- MAIN INTERFACE ---
st.title("🔍 Real-Time AI Object Detection")
st.write("This dashboard uses **YOLOv8** to process video frames and identify objects in real-time.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🖥️ Live Feed")
    # Fixed: Creating a local placeholder image so no broken icon appears
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Statistics")
    fps_metric = st.empty()
    count_metric = st.empty()
    
    st.markdown("### 📝 Instructions")
    st.info("""
    1. Select model size from the sidebar.
    2. Toggle the **Start Live Detection** switch.
    3. Use the slider to filter out weak detections.
    4. View live processing speed and object counts below.
    """)

# --- AI LOGIC ---
model = load_yolo_model(model_size)

if run_webcam:
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while cap.isOpened() and run_webcam:
        success, frame = cap.read()
        if not success:
            st.error("Error: Could not access the webcam.")
            break

        # Run AI Inference
        # stream=True makes it memory efficient for video
        results = model(frame, conf=confidence, stream=True)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Process and Annotate
        for r in results:
            annotated_frame = r.plot()  # Draw boxes and labels
            obj_count = len(r.boxes)    # Count detections
            
        # Convert BGR (OpenCV) to RGB (Streamlit)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Update Web UI
        frame_placeholder.image(annotated_frame, channels="RGB")
        fps_metric.metric("Processing Speed", f"{int(fps)} FPS")
        count_metric.metric("Objects in View", f"{obj_count}")

    cap.release()
else:
    # Fix: Create a nice grey "Webcam Off" box if not running
    # This prevents the "broken image" icon you saw earlier
    idle_img = np.zeros((480, 640, 3), dtype=np.uint8) + 40 
    cv2.putText(idle_img, "Webcam is Idle. Flip toggle to start.", (120, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    frame_placeholder.image(idle_img, channels="RGB")
