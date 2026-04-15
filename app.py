import cv2
import os
import json
import time
import threading
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
from engine.detector import FaceDetector
from engine.recognizer import FaceRecognizer
from engine.trainer import FaceTrainer
from core.storage import StorageManager
from core.scheduler import AttendanceScheduler

app = Flask(__name__)
CORS(app)

# ── Initialise core components ──────────────────────────────────────────────
storage   = StorageManager()
scheduler = AttendanceScheduler(storage)
scheduler.start()

detector   = FaceDetector()
recognizer = FaceRecognizer()
trainer    = FaceTrainer()

DATASETS_DIR   = 'datasets'
PERSONNEL_FILE = 'personnel_meta.json'

# ── Personnel Metadata Helpers ──────────────────────────────────────────────
def load_personnel():
    if os.path.exists(PERSONNEL_FILE):
        with open(PERSONNEL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_personnel(data):
    with open(PERSONNEL_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ── Enrollment State ─────────────────────────────────────────────────────────
enroll_state = {
    'active':   False,
    'name':     '',
    'captured': 0,
    'target':   60,
    'cap':      None,
    'thread':   None,
}
enroll_lock = threading.Lock()


def _enroll_loop():
    global enroll_state
    name    = enroll_state['name']
    folder  = os.path.join(DATASETS_DIR, name)
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    with enroll_lock:
        enroll_state['cap']           = cap
        enroll_state['multi_face_warn'] = False

    count = 0
    while enroll_state['active'] and count < enroll_state['target']:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        faces = detector.detect(frame)

        # ── Only capture when EXACTLY ONE face is visible ──────────
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
            filename = os.path.join(folder, f'{count+1}.png')
            cv2.imwrite(filename, face)
            count += 1
            with enroll_lock:
                enroll_state['captured']        = count
                enroll_state['multi_face_warn'] = False
        elif len(faces) > 1:
            with enroll_lock:
                enroll_state['multi_face_warn'] = True  # Signal UI to warn
        else:
            with enroll_lock:
                enroll_state['multi_face_warn'] = False

        time.sleep(0.05)

    cap.release()
    with enroll_lock:
        enroll_state['active'] = False
        enroll_state['cap']    = None


# ── Video streamer ───────────────────────────────────────────────────────────
class VideoStreamer:
    def __init__(self):
        self.cap     = None
        self.running = False
        self._lock   = threading.Lock()
        self._frame  = None
        self._thread = None

    def start(self):
        if self.running:
            return
        self.cap     = cv2.VideoCapture(0)
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            faces = detector.detect(frame)
            for (x, y, w, h) in faces:
                name, conf = recognizer.recognize(frame, (x, y, w, h))
                color = (34, 197, 94) if name != "Unknown" else (239, 68, 68)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name}  {conf}%" if name != "Unknown" else "Unknown"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
                cv2.putText(frame, label, (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if name != "Unknown" and conf > 20:
                    storage.mark_presence(name)

            with self._lock:
                self._frame = frame.copy()
            time.sleep(0.03)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


streamer = VideoStreamer()


def gen_mjpeg():
    while True:
        frame = streamer.get_frame()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)


# ── Enrollment stream ─────────────────────────────────────────────────────────
def gen_enroll_mjpeg():
    """Preview stream for the enrollment wizard (draws face box + progress)."""
    while enroll_state['active']:
        cap = enroll_state.get('cap')
        if cap and cap.isOpened():
            ok, frame = cap.read()
            if ok:
                faces = detector.detect(frame)
                for (x, y, w, h) in faces:
                    captured = enroll_state['captured']
                    target   = enroll_state['target']
                    pct      = int(captured / target * 100)
                    color    = (99, 102, 241)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"Capturing  {captured}/{target}  ({pct}%)"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    streamer.start()
    return jsonify({'status': 'started'})

@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    streamer.stop()
    return jsonify({'status': 'stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance')
def attendance():
    """Return attendance for a given date (default: today)."""
    date = request.args.get('date', '')
    return jsonify(storage.get_by_date(date) if date else storage.get_today())


@app.route('/api/attendance/dates')
def attendance_dates():
    """Return all unique dates that have attendance records."""
    import sqlite3
    conn = sqlite3.connect('history.db')
    cur  = conn.execute('SELECT DISTINCT date FROM attendance ORDER BY date DESC')
    dates = [r[0] for r in cur.fetchall()]
    conn.close()
    return jsonify(dates)


@app.route('/api/attendance/delete', methods=['POST'])
def delete_attendance():
    """Delete a single attendance record by name+date."""
    import sqlite3
    data = request.json or {}
    name = data.get('name', '').strip()
    date = data.get('date', '')
    if not name or not date:
        return jsonify({'success': False, 'message': 'name and date required'}), 400
    conn = sqlite3.connect('history.db')
    conn.execute('DELETE FROM attendance WHERE name=? AND date=?', (name, date))
    conn.commit()
    conn.close()
    storage.rebuild_excel()   # keep Excel in sync
    return jsonify({'success': True, 'message': f'Deleted {name} for {date}'})


@app.route('/api/attendance/clear-today', methods=['POST'])
def clear_today():
    """Wipe ALL attendance records for today."""
    import sqlite3
    from datetime import datetime
    date = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect('history.db')
    conn.execute('DELETE FROM attendance WHERE date=?', (date,))
    conn.commit()
    conn.close()
    storage.rebuild_excel()   # keep Excel in sync
    return jsonify({'success': True, 'message': f"All records for {date} cleared."})


@app.route('/api/stats')
def stats():
    return jsonify(storage.get_stats())

@app.route('/api/users')
def users():
    """Returns list of {name, id, department, role} from personnel metadata."""
    meta  = load_personnel()
    dirs  = storage.get_users()
    result = []
    for name in dirs:
        info = meta.get(name, {})
        result.append({
            'name':       name,
            'pid':        info.get('pid', '—'),
            'department': info.get('department', '—'),
            'role':       info.get('role', '—'),
            'email':      info.get('email', '—'),
        })
    return jsonify(result)

@app.route('/api/train', methods=['POST'])
def train():
    ok, msg = trainer.train()
    if ok:
        recognizer.reload()
    return jsonify({'success': ok, 'message': msg})

@app.route('/api/mark-absent', methods=['POST'])
def mark_absent():
    count = storage.mark_absentees()
    return jsonify({'message': f'Marked absentees for {count} registered users.'})

@app.route('/api/cutoff', methods=['POST'])
def set_cutoff():
    data   = request.json or {}
    hour   = int(data.get('hour', 9))
    minute = int(data.get('minute', 0))
    scheduler.set_cutoff(hour, minute)
    return jsonify({'message': f'Cutoff set to {hour:02}:{minute:02}'})

@app.route('/api/export')
def export_excel():
    from datetime import datetime
    if os.path.exists('attendance.xlsx'):
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename  = f'attendance_{date_str}.xlsx'
        return send_file('attendance.xlsx', as_attachment=True, download_name=filename)
    return jsonify({'error': 'No attendance file found'}), 404


@app.route('/api/export/<date>')
def export_by_date(date):
    """Export attendance for a specific date as Excel."""
    import sqlite3, io
    import pandas as pd
    conn = sqlite3.connect('history.db')
    df   = pd.read_sql_query(
        'SELECT name AS Name, date AS Date, time AS Time, status AS Status '
        'FROM attendance WHERE date=? ORDER BY time',
        conn, params=(date,)
    )
    conn.close()
    if df.empty:
        return jsonify({'error': f'No records found for {date}'}), 404
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    filename = f'attendance_{date}.xlsx'
    return send_file(buf, as_attachment=True, download_name=filename,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# ── Enrollment Endpoints ───────────────────────────────────────────────────
@app.route('/api/enroll/start', methods=['POST'])
def enroll_start():
    global enroll_state
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': 'Name is required.'}), 400

    # Stop live stream if running (share camera)
    streamer.stop()
    time.sleep(0.3)

    with enroll_lock:
        if enroll_state['active']:
            return jsonify({'success': False, 'message': 'Enrollment already in progress.'}), 409
        enroll_state.update({'active': True, 'name': name, 'captured': 0, 'target': 60})

    t = threading.Thread(target=_enroll_loop, daemon=True)
    with enroll_lock:
        enroll_state['thread'] = t
    t.start()

    # Save metadata
    meta = load_personnel()
    meta[name] = {
        'pid':        data.get('pid', ''),
        'department': data.get('department', ''),
        'role':       data.get('role', ''),
        'email':      data.get('email', ''),
    }
    save_personnel(meta)

    return jsonify({'success': True, 'message': f'Enrollment started for {name}.'})


@app.route('/api/enroll/status')
def enroll_status():
    with enroll_lock:
        return jsonify({
            'active':         enroll_state['active'],
            'name':           enroll_state['name'],
            'captured':       enroll_state['captured'],
            'target':         enroll_state['target'],
            'multi_face_warn': enroll_state.get('multi_face_warn', False),
        })


# (enroll_stop is defined above, near enroll_feed)


@app.route('/enroll_feed')
def enroll_feed():
    return Response(gen_enroll_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/enroll/stop', methods=['POST'])
def enroll_stop():
    import shutil
    data   = request.json or {}
    delete = data.get('delete_partial', False)
    name   = ''
    with enroll_lock:
        enroll_state['active'] = False
        name = enroll_state.get('name', '')
    # If caller asks to wipe partial data (Cancel case)
    if delete and name:
        folder = os.path.join(DATASETS_DIR, name)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        # Remove metadata too
        meta = load_personnel()
        meta.pop(name, None)
        save_personnel(meta)
    with enroll_lock:
        enroll_state['name']     = ''
        enroll_state['captured'] = 0
    return jsonify({'success': True})


@app.route('/api/personnel/delete', methods=['POST'])
def delete_personnel():
    data = request.json or {}
    name = data.get('name', '').strip()
    import shutil
    folder = os.path.join(DATASETS_DIR, name)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    meta = load_personnel()
    meta.pop(name, None)
    save_personnel(meta)
    return jsonify({'success': True, 'message': f'{name} removed.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
