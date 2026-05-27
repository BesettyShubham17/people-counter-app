import os
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, session
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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            flash('No file part in request')
            return redirect(url_for('home'))

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('home'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run YOLO inference
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

        flash('Invalid file type')
        return redirect(url_for('home'))

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        flash(f'Processing error: {str(e)}')
        return redirect(url_for('home'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    return upload_file()

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)