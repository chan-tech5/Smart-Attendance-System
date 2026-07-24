# Smart Attendance System — AI Face Recognition

An enterprise-grade, real-time face recognition attendance platform built with Python (Flask + OpenCV) and a full-featured, responsive web dashboard.

## Features
- **AI Recognition**: Dual-model pipeline — FisherFace (primary) + LBPH (fallback)
- **Live Browser Dashboard**: Real-time video feed with name overlays, no separate frontend build needed
- **Light & Dark Mode**: Persistent theme toggling across all pages with a unified design system and visual consistency
- **Historical Attendance Navigation**: Browse past attendance records by date, with smart labels ("Today", "Yesterday") and Sunday exclusions
- **Face Enrollment Wizard**: Register students/employees from the browser — enter their ID, department, role, and capture 60 face samples via live webcam
- **Smart Capture**: Rejects frames with 0 or multiple people — only captures when exactly 1 face is visible
- **Automated Logging**: Every check-in is instantly saved to a local SQLite database (`history.db`) serving as the single source of truth
- **Dynamic Excel Export**: Download the attendance sheet for any selected date directly from the dashboard on-the-fly (`attendance_<date>.xlsx`)
- **Auto Absentee Marking**: Set a daily cutoff time (default 9:00 AM) — the system marks everyone who hasn't checked in as Absent automatically
- **Personnel Management**: View, delete, and manage all enrolled subjects directly from the browser interface

## Tech Stack
| Layer | Technology |
|---|---|
| AI Engine | OpenCV (FisherFace + LBPH), Haar Cascade |
| Web Server | Python Flask |
| Storage | SQLite (`history.db`) |
| Export | Pandas + OpenPyXL (Dynamic Excel Generation) |
| Frontend | HTML, Tailwind CSS (CDN), Vanilla JS |
| Task Scheduler | APScheduler (Background job for daily cutoff) |

## Project Structure
```
Smart-Attendance-System/
|
|-- app.py                              # Main Flask server
|
|-- engine/                             # AI Recognition Architecture
|   |-- detector.py                     # Haar Cascade face detection
|   |-- recognizer.py                   # FisherFace + LBPH recognition
|   +-- trainer.py                      # Model training from face samples
|
|-- core/                               # Business Logic
|   |-- storage.py                      # SQLite manager & data querying
|   +-- scheduler.py                    # Background daily absentee scheduler
|
|-- templates/
|   +-- index.html                      # Full dashboard (SPA, responsive, theming)
|
|-- datasets/                           # Face image samples (auto-created)
|   +-- <PersonName>/                   # 60 grayscale images per person
|
|-- haarcascade_frontalface_default.xml  # OpenCV Haar Cascade model
|-- history.db                          # Primary SQLite attendance database
|-- personnel_meta.json                 # Enrolled personnel metadata
|-- fisher_model.yml                    # Trained FisherFace model
|-- names.pkl                           # Name-to-ID mapping dictionary
+-- requirements.txt                    # Python dependencies
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python app.py
```

### 3. Open the Dashboard
Visit [http://localhost:5000](http://localhost:5000) in your browser.

## Workflow

### 1. Registering a New Person
- Go to the **Personnel** tab → click **"Enroll New Person"**
- Fill in: Name, ID/Roll No., Department, Role, Email
- Click **"Continue to Face Capture"** — the webcam starts
- System captures **60 face samples** automatically (1 face at a time)
- After capture, click **"Re-Train Model"** to activate the person in the AI system

### 2. Running Attendance
- Go to the **Live Scanner** tab → click **"Start Attendance Scan"**
- The AI recognizes faces and logs them instantly to the database
- You can view today's logs on the right side under *Recent Detections*

### 3. Reviewing Records & Exporting
- View historical records in the **Attendance Records** tab
- Select a specific date from the date picker 
- Click **"Download Excel"** to instantly export an `attendance_<date>.xlsx` file

### 4. Auto Absentee Marking
- Navigate to the **Settings** tab → set your daily cutoff time (e.g., 09:00)
- The system checks in the background — anyone not logged for the day will be marked **Absent** automatically
- You can also force trigger it with the **"Mark Everyone Else Absent Now"** button

