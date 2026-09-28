# Intelligent Video Surveillance System

An enterprise-grade, real-time video surveillance system featuring an optimized AI inference pipeline (YOLOv8 + DeepSORT), motion detection pre-filtering, and a responsive React dashboard. 

This platform is designed to process live video feeds, detect and track suspicious activities (like unauthorized vehicles or persons), and alert security personnel instantly via a sleek web interface.

---

## 🌟 Key Features

*   **Optimized Inference Pipeline:** Uses OpenCV background subtraction (MOG2) to ignore static frames, reserving heavy YOLOv8 ONNX processing only for when motion is detected.
*   **Multi-Object Tracking:** Integrates DeepSORT to assign unique IDs to targets and track them across frames, even through temporary occlusions.
*   **FastAPI Backend:** A high-performance REST API that acts as the orchestration layer, logging events to a SQLite/PostgreSQL database via SQLAlchemy.
*   **Modern React Dashboard:** A dark-mode, glassmorphism-styled web app that streams live camera data, overlays tracking boxes, and displays instant toast alerts for security breaches.
*   **Production-Ready Structure:** Clean separation of concerns with a modular directory layout.

---

## 📁 Architecture

```text
intelligent-surveillance/
│
├── backend/
│   ├── app.py                # FastAPI entry point
│   ├── models/               # Contains yolov8n.onnx AI weights
│   ├── detection/            # Core ML (Motion, YOLO, DeepSORT, Preprocessing)
│   ├── database/             # SQLite DB and SQLAlchemy ORM logic
│   ├── utils/                # Global config paths and helper functions
│   └── requirements.txt      # Python dependencies
│
└── frontend/                 # React.js application
    ├── src/
    │   ├── components/       # VideoFeed, EventLog, Alerts
    │   ├── pages/            # Future scalable routes
    │   ├── App.jsx           # Global layout & state
    │   └── index.css         # Custom UI Design System
    └── package.json          # Node dependencies
```

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.8+**
*   **Node.js & npm**
*   Webcam (for live testing)

### 1. Backend Setup

First, navigate to the root directory and set up the Python virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Navigate to backend and install dependencies
cd backend
pip install -r requirements.txt
```

**Note on YOLOv8:** The system requires the ONNX-exported YOLOv8 model. You can generate it by running:
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
# Move the resulting yolov8n.onnx to backend/models/
```

### 2. Frontend Setup

Open a new terminal window, navigate to the frontend directory, and install dependencies:

```bash
cd frontend
npm install
```

---

## 🖥️ Running the Application

You need to run both the backend and frontend simultaneously.

**Terminal 1 (Backend API)**
```bash
# Ensure venv is activated!
cd backend
python app.py
```
*The API will start at `http://localhost:8000`*

**Terminal 2 (Frontend Dashboard)**
```bash
cd frontend
npm run dev
```
*The dashboard will start at `http://localhost:5173`. Open this URL in your browser and allow camera permissions.*

---

## 🔌 API Endpoints

The FastAPI backend provides interactive documentation. Once running, visit `http://localhost:8000/docs`.

*   **`POST /api/v1/detect`**: Submit a video frame (multipart form data). Returns bounding boxes, classes, and track IDs.
*   **`GET /api/v1/events`**: Fetch paginated historical event logs from the database.
*   **`GET /api/v1/alerts`**: Poll for recent high-severity security breaches.

---

## 🛠️ Built With
*   **Backend:** FastAPI, OpenCV, ONNX Runtime, DeepSORT, SQLAlchemy
*   **Frontend:** React, Vite, Lucide Icons, React Hot Toast
