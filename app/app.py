from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory 
from ultralytics import YOLO
import cv2
import os
import time
import logging
import numpy as np
from threading import Lock

# -*- coding: utf-8 -*-

app = Flask(__name__)

##################### model #####################
yolo = YOLO('model/yolo11x.pt')
model = YOLO('model/Iteam_object.pt')

##################### rtsp link #####################
live_link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

##################### make path #####################
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/result', exist_ok=True)

def cleanup_old_files():
    """定期清理舊的上傳和結果檔案"""
    upload_dir = os.path.join('static', 'uploads')
    result_dir = os.path.join('static', 'result')
    current_time = time.time()
    for directory in [upload_dir, result_dir]:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) < current_time - 86400:
                os.remove(filepath)

##################### allow upload file type #####################
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

##################### websit page #####################
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')

##################### live  #####################

def live_detect(live_link):
    cap = cv2.VideoCapture(live_link)
    
    try:
        if not cap.isOpened():
            raise RuntimeError("無法打開攝像頭")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(source=frame, classes=[0,2,7], conf=0.8)
       
            display_frame = result[0].plot() if len(result[0].boxes) > 0 else frame
            
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                break
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        cap.release()

@app.route("/upload")
def image_video_predict():
    return render_template('upload_web.html')

def live_detect(live_link):
    cap = cv2.VideoCapture(live_link)
    
    def create_no_truck_image():
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No Truck Detected"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
        cv2.putText(img, text, (textX, textY), font, 1, (0, 0, 0), 2)
        return img

    try:
        if not cap.isOpened():
            raise RuntimeError("無法打開攝像頭")

        no_truck_image = create_no_truck_image()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = yolo(source=frame, classes=[0,2,7], conf=0.8)
            
            if len(result[0].boxes) > 0:
                display_frame = result[0].plot()
            else:
                display_frame = no_truck_image
            
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                break
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        cap.release()

@app.route('/live-detect')
def LiveDetect():
    detected_items = []
    cap = cv2.VideoCapture(live_link)
    
    try:
        ret, frame = cap.read()
        if ret:
            results = yolo(source=frame, classes=[0,2,7], conf=0.8)
            if results and len(results[0].boxes) > 0:
                detected_items = [results[0].names[int(box.cls)] for box in results[0].boxes]
    finally:
        cap.release() 
    
    return jsonify(detected_items=detected_items)

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(live_detect(live_link),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

##################### image and video #####################
@app.route("/upload", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return "未選擇文件", 400
        
        f = request.files['file']
        if f.filename == '':
            return "未選擇文件", 400
        
        if not allowed_file(f.filename):
            return "不支援的文件類型", 400

        filename = f.filename
        filepath = os.path.join('static', 'uploads', filename)
        f.save(filepath)

        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension in ['jpeg', 'jpg', 'png']:##################### image #####################
            img = cv2.imread(filepath)
            
            results_paths = []
        
            for model_name in ['model']:
                current_model = globals()[model_name]
                detections = model(img, conf=0.8)
                result = detections[0].plot()
                
                result_filename = f"{os.path.splitext(f.filename)[0]}_{model_name}.jpg"
                result_path = os.path.join('static', 'result', result_filename)
                
                cv2.imwrite(result_path, result)
                results_paths.append(result_path)
            
            return render_template('upload_web.html', image_url=result_path)

            
        elif file_extension == 'mp4': ##################### video #####################
            cap = cv2.VideoCapture(filepath)
            
            try:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                output_path = os.path.join('static', 'result', 'output.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    
                    results = model(gray_frame_3ch)
                    annotated_frame = results[0].plot()
                    
                    out.write(annotated_frame)
                
                return redirect(f'/video_feed/{output_path}')
            
            finally:
                cap.release()
                out.release()
    
    return render_template('upload_web.html')

##################### mp4 #####################
def get_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    finally:
        cap.release()
        
@app.route('/video_feed')
def video_feed(video_path):
    return Response(get_frame(video_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
