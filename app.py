import os
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime

torch.set_grad_enabled(False)

# ─── Suppress OpenCV GUI / libxcb crash on headless servers ──────────────────
os.environ["ULTRALYTICS_NO_OPENCV"] = "1"
os.environ["YOLO_OPENCV"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'crowd-estimator-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

db = SQLAlchemy(app)

# User model for database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Ensure upload folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ─── Model Loading (lazy — avoids import crash on headless servers) ──────────
_model = YOLO("yolov8n.pt")

def get_model():
    return _model

# ─── Helper Functions ────────────────────────────────────────────────────────

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS

def process_image(img):
    model = get_model()
    with torch.no_grad():
        results = model(img)
    return len(results[0].boxes)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Successfully registered! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return {"error": "No file part in request"}, 400

        file = request.files['file']
        if file.filename == '':
            return {"error": "No selected file"}, 400

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if ext in ALLOWED_VIDEO_EXTENSIONS:
                # Render the CCTV interface which will call /video_feed
                return render_template('cctv.html', filename=filename)

            else:
                # Run YOLO inference on Image
                img = Image.open(filepath).convert("RGB")
                model = get_model()
                with torch.no_grad():
                    results = model(img)

                boxes = results[0].boxes
                people_count = len(boxes)

                # Draw bounding boxes using PIL
                draw = ImageDraw.Draw(img)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    # Green bounding box
                    draw.rectangle([(x1, y1), (x2, y2)], outline='#00FF00', width=3)
                    draw.text((x1, max(y1 - 14, 0)), f'{conf:.0%}', fill='#00FF00')

                # Save annotated image
                output_filename = f'annotated_{filename}'
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                img.save(output_path)

                return render_template('result.html',
                                       filename=output_filename,
                                       original_filename=filename,
                                       people_count=people_count)

        return {"error": "Invalid file type"}, 400

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        flash(f'Processing error: {str(e)}')
        return redirect(url_for('home'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    return upload_file()

def generate_frames(filepath):
    import cv2
    import time
    
    cap = cv2.VideoCapture(filepath)
    model = get_model()
    prev_time = time.time()
    CROWD_THRESHOLD = 15
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        with torch.no_grad():
            # Use YOLO object tracking
            results = model.track(frame, persist=True, classes=[0], verbose=False)
            
        boxes = results[0].boxes
        count = len(boxes)
        
        confs = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        avg_conf = (sum(confs) / len(confs)) * 100 if len(confs) > 0 else 0
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            # Draw tracking box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{track_id} {conf:.0%}" if track_id != -1 else f"{conf:.0%}"
            cv2.putText(frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        
        # Overlay CCTV dashboard text
        cv2.putText(frame, f"People Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {avg_conf:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if count > CROWD_THRESHOLD:
            cv2.putText(frame, "ALERT: CROWD LIMIT EXCEEDED!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return "File not found", 404
    return Response(generate_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)