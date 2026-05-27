# app.py
import os
import torch
torch.set_grad_enabled(False)

# ─── Suppress OpenCV GUI / libxcb crash on headless servers ──────────────────
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime

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

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Model Loading (lazy — avoids import crash on headless servers) ──────────
_model_s = None
_model_l = None

def get_model_small():
    global _model_s
    if _model_s is None:
        from ultralytics import YOLO
        _model_s = YOLO("yolov5s.pt")
    return _model_s

def get_model_large():
    global _model_l
    if _model_l is None:
        from ultralytics import YOLO
        _model_l = YOLO("yolov5l.pt")
    return _model_l

# ─── Helper Functions ────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def count_people(image_path):
    """Runs YOLOv5l on a still image, returns (count, detections_tensor)."""
    model = get_model_large()
    results = model(image_path, imgsz=1280, conf=0.25, iou=0.4, classes=[0], verbose=False)
    boxes = results[0].boxes
    return len(boxes), boxes

def draw_box_plot(frame, count_history, max_history=100):
    import cv2
    height, width = frame.shape[:2]
    plot_height = 100
    plot_y = height - plot_height - 10
    plot_x = 50
    plot_width = width - 100

    cv2.rectangle(frame, (plot_x - 2, plot_y - 2),
                  (plot_x + plot_width + 2, plot_y + plot_height + 2),
                  (255, 215, 0), 2)

    display_history = count_history[-max_history:] if len(count_history) > max_history else count_history

    if display_history:
        max_count = max(max(display_history), 1)
        for i, count in enumerate(display_history):
            x = plot_x + int((i / max_history) * plot_width)
            y = plot_y + plot_height - int((count / max_count) * plot_height)
            cv2.circle(frame, (x, y), 3, (0, 215, 255), -1)

        current_count = display_history[-1]
        cv2.putText(frame, f'Current Count: {current_count}',
                    (width - 320, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 215, 255), 2)
        cv2.putText(frame, str(max_count), (plot_x - 40, plot_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, '0', (plot_x - 20, plot_y + plot_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def process_video(video_path):
    import cv2
    """Process video frame-by-frame using YOLOv5l via ultralytics."""
    model = get_model_large()

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'counted_{ts}.mp4'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    # Use mp4v for broad compatibility (works without ffmpeg)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    frame_results = []
    count_history = []

    # Analyse 2 frames per second for speed
    process_every_n = max(1, int(fps // 2))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % process_every_n == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, imgsz=1280, conf=0.05, iou=0.45,
                            classes=[0], verbose=False)
            boxes = results[0].boxes
            people_count = len(boxes)
            count_history.append(people_count)

            seconds_elapsed = frame_number / fps
            minutes = int(seconds_elapsed // 60)
            secs = int(seconds_elapsed % 60)
            frame_results.append({
                'frame': frame_number,
                'timestamp': f'{minutes:02d}:{secs:02d}',
                'count': people_count
            })

            # Draw bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, max(y1 - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

            frame = draw_box_plot(frame, count_history)
            avg = sum(count_history) / len(count_history)
            cv2.putText(frame, f'Avg: {avg:.1f}', (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 2)
        else:
            if count_history:
                frame = draw_box_plot(frame, count_history)

        out.write(frame)

    video.release()
    out.release()
    return output_path, frame_results

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
            flash('No file part')
            return redirect(url_for('home'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('home'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            people_count, boxes = count_people(filepath)

            img = Image.open(filepath).convert("RGB")
            draw = ImageDraw.Draw(img)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=3)
                draw.text((x1, max(y1 - 12, 0)), f'{conf:.2f}', fill='red')

            output_filename = f'annotated_{filename}'
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            img.save(output_filepath)

            return render_template('result.html',
                                   filename=output_filename,
                                   original_filename=filename,
                                   people_count=people_count)
        flash('Invalid file type')
        return redirect(url_for('home'))
    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}, 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No video file provided')
        return redirect(url_for('home'))
    file = request.files['video']
    if file.filename == '':
        flash('No selected video')
        return redirect(url_for('home'))
    if file and allowed_video(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        output_path, frame_results = process_video(filepath)
        return render_template('video_result.html',
                               video_filename=os.path.basename(output_path),
                               original_filename=filename,
                               frame_results=frame_results)
    flash('Invalid video file type')
    return redirect(url_for('home'))

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)