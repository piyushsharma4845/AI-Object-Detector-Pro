# 🔍 AI Vision Pro: Real-Time Object Detection Dashboard


An end-to-end Computer Vision application that performs real-time object detection using the **YOLOv8 (You Only Look Once)** architecture. Built with a professional Streamlit dashboard, this project demonstrates the integration of deep learning models into a user-friendly web interface.

---

## 🚀 Live Demo
**Access the live application here:** [👉 Click to Open App](YOUR_LIVE_LINK_HERE)

---

## 🌟 Key Features
- **Real-Time Detection:** Processes live video stream with low latency.
- **Dynamic Model Switching:** Toggle between YOLOv8 'Nano' (Speed) and 'Small' (Accuracy) versions.
- **Interactive Confidence Control:** A real-time slider to adjust the model's sensitivity (Precision vs. Recall).
- **Live Performance Metrics:** Real-time FPS (Frames Per Second) and object counter.
- **Educational UI:** Integrated explanations on how AI confidence thresholds work.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Deep Learning** | Ultralytics YOLOv8 |
| **Frontend UI** | Streamlit |
| **Computer Vision** | OpenCV |
| **Data Processing** | NumPy |
| **Deployment** | Streamlit Community Cloud |

---

## 🎓 Core AI Concept: Confidence Thresholding
As a Computer Science Engineering student, this project focuses on the balance between **Precision** and **Recall**:
- **Confidence Threshold:** The minimum probability required for the model to "believe" an object exists. 
- **The Trade-off:** Lowering the threshold increases **Recall** (finding more objects) but may lead to **False Positives** (guessing wrong). Increasing it improves **Precision** (high accuracy) but might miss smaller or partially hidden objects.

---
