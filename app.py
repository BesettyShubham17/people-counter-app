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
_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model

# ─── Helper Functions ────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(path):
    model = get_model()
    results = model(path)
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
        file = request.files['file']
        path = "uploads/" + file.filename
        file.save(path)

        result = process_image(path)

        return {"success": True, "count": result}

    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    return upload_file()

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)