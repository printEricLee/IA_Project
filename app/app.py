from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory 
from ultralytics import YOLO
import cv2
import os
import time
import logging
import os
import logging
import numpy as np

# -*- coding: utf-8 -*-

############################
# f is the file you upload #
############################

app = Flask(__name__)

# model_yolov8 = YOLO('model/Iteam_object.pt')

model_yolov5 = YOLO('model/For_video.pt')

model_yolov8_2 = YOLO('model/wet_dry.pt')

model_yolov8 = YOLO('model/yolo11x.pt')

live_link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/result', exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

############### image ###############
@app.route("/upload", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400
        
        if allowed_file(f.filename):
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'static', 'uploads', f.filename)
            f.save(filepath)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}
                                                
            if f.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                file_extension = f.filename.rsplit('.', 1)[1].lower()
            
            if file_extension in ['jpeg', 'jpg', 'png']:
                img = cv2.imread(filepath)
                models = ['model_yolov5', 'model_yolov8']
                results_paths = []

                for model_name in models:
                    model = globals()[model_name]
                    detections = model(img)
                    result = detections[0].plot()
                    
                    result_filename = f"{os.path.splitext(f.filename)[0]}_{model_name}.jpg"
                    result_path = os.path.join('static', 'result', result_filename)
                    
                    cv2.imwrite(result_path, result)
                    results_paths.append(result_path)

            elif file_extension == 'mp4': 
                    video_path = filepath
                    cap = cv2.VideoCapture(video_path)

                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break                                                      

                        results_car_object = model_yolov5(frame)
                        results_wet_dry = model_yolov8_2(frame)
                        print(results_car_object)
                        cv2.waitKey(1)

                        res_plotted = results_car_object[0].plot()

                    cap.release()
                    out.release()

                    return videos_feed()

     
        return render_template('upload_web.html', image_url=result_path)
    
    else:

        return render_template('upload_web.html')

############### video ###############
detected_items = []

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)

@app.route('/videos_feed/<path:video_path>')
def videos_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolov5(frame)
        if results:
            detected_items = [results[0].names[int(box.cls)] for box in results[0].boxes]
            frame_bytes = detected_items.tobytes()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    
    return jsonify(detected_items=detected_items)

############### live ###############
def live_detect(live_link):
    cap = cv2.VideoCapture(live_link)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        result = model_yolov5(frame)

        annotated_frame = result[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("无法编码视频帧")
            break
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/live-detect')
def LiveDetect():
    detected_items = []
    cap = cv2.VideoCapture(live_link)
    
    try:
        ret, frame = cap.read()
        if ret:
            results = model_yolov5(frame)
            if results and hasattr(results[0], 'boxes'):
                detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data]
    finally:
        cap.release() 
    return jsonify(detected_items=detected_items)

# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(live_detect(live_link), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)