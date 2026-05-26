# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
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

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_people(image_path):
    # Load a larger YOLOv5 model for better accuracy
    global model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Changed from yolov5s to yolov5l
    
    # Load and process image
    img = Image.open(image_path)
    
    # Increase inference size significantly for better detection
    results = model(img, size=1280)  # Increased from 640 to 1280
    
    # Lower confidence threshold to detect more people
    confidence_threshold = 0.25  # Lowered from 0.3
    iou_threshold = 0.4  # Lowered for less aggressive NMS
    
    # Get predictions and filter for person class
    predictions = results.pred[0]
    people_detections = predictions[predictions[:, -1] == 0]
    
    # Filter by confidence
    confident_detections = people_detections[people_detections[:, 4] >= confidence_threshold]
    
    # Apply Non-Maximum Suppression
    nms_indices = torch.ops.torchvision.nms(
        confident_detections[:, :4],
        confident_detections[:, 4],
        iou_threshold
    )
    
    final_detections = confident_detections[nms_indices]
    
    return len(final_detections), final_detections

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
        
        # Count people and get detections
        people_count, detections = count_people(filepath)
        
        # Draw bounding boxes on the image
        img = Image.open(filepath)
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            
            # Draw rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
            # Add confidence score
            draw.text((x1, y1-10), f'{conf:.2f}', fill='red')
        
        # Save annotated image
        output_filename = f'annotated_{filename}'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        img_draw.save(output_filepath)
        
        return render_template('result.html', 
                             filename=output_filename,
                             original_filename=filename,
                             people_count=people_count)
    
    flash('Invalid file type')
    return redirect(url_for('home'))

def draw_box_plot(frame, count_history, max_history=100):
    height, width = frame.shape[:2]
    
    plot_height = 100
    plot_y = height - plot_height - 10
    plot_x = 50
    plot_width = width - 100
    
    cv2.rectangle(frame, (plot_x-2, plot_y-2), 
                 (plot_x+plot_width+2, plot_y+plot_height+2), 
                 (255, 255, 255), 2)
    
    if len(count_history) > max_history:
        count_history = count_history[-max_history:]
    
    if len(count_history) > 0:
        max_count = max(max(count_history), 1)
        
        for i, count in enumerate(count_history):
            x = plot_x + int((i / max_history) * plot_width)
            y = plot_y + plot_height - int((count / max_count) * plot_height)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        current_count = count_history[-1]
        cv2.putText(frame, f'Current Count: {current_count}', 
                   (width-300, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        cv2.putText(frame, str(max_count), (plot_x-40, plot_y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, '0', (plot_x-20, plot_y+plot_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def process_video(video_path):
    global model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Use yolov5l for better detection
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'counted_{timestamp}.webm'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # VP09 codec for webm which works well in browsers
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    frame_results = []
    count_history = []
    
    # Process 2 frames per second to be fast
    process_every_n_frames = max(1, int(fps // 2))
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        frame_number += 1
        
        if frame_number % process_every_n_frames == 0:
            # Convert BGR to RGB for model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, size=1280)
            
            predictions = results.pred[0]
            # Class 0 is person
            people_detections = predictions[predictions[:, -1] == 0]
            confident_detections = people_detections[people_detections[:, 4] >= 0.05]
            
            people_count = len(confident_detections)
            count_history.append(people_count)
            
            timestamp_seconds = frame_number / fps
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            frame_results.append({
                'frame': frame_number,
                'timestamp': f'{minutes:02d}:{seconds:02d}',
                'count': people_count
            })
            
            # Draw boxes
            for det in confident_detections:
                x1, y1, x2, y2 = map(int, det[:4])
                conf = det[4].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            frame = draw_box_plot(frame, count_history)
            
            avg_count = sum(count_history) / len(count_history)
            cv2.putText(frame, f'Avg Count: {avg_count:.1f}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        else:
            if count_history:
                frame = draw_box_plot(frame, count_history)
            out.write(frame)
            
    video.release()
    out.release()
    
    return output_path, frame_results

def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)