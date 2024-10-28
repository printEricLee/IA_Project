from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory 
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import uuid
import threading
import time
import subprocess
import logging

# f is the file name 

app = Flask(__name__)

model_yolov8 = YOLO('model/Iteam_object.pt')

model_yolov5 = YOLO('model/For_video.pt')

model_yolov8_2 = YOLO('model/wet_dry.pt')

link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

base_dir = os.path.join('static', 'runs')
os.make.path(base_dir,'output')

############### give every image name ###############
# def generate_unique_filename(filename):
#     _, extension = os.path.splitext(filename)
#     unique_filename = str(uuid.uuid4()) + extension
#     return unique_filename
####################################################

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')
    
@app.route("/uploadVideo")
def image_video_predict():
    return render_template('UploadVideo.html')

############### image ###############
@app.route('/upload', methods=['GET', 'POST'])
def predict_img():
    if request.method == 'POST':
        if 'image' not in request.files:
            # make file and upload image or video in it
            os.makedirs('uploads', exist_ok=True)
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            f.save(filepath)

        ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}
                                               
        if f.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            file_extension = f.filename.rsplit('.', 1)[1].lower()
        
        if f.filename == '':
            return redirect(request.url)

        if f:
            unique_filename = generate_unique_filename(f.filename)
            original_image_path = os.path.join(base_dir, 'originals', unique_filename)
            f.save(original_image_path)

            # Model 1
            results1 = model_yolov8(original_image_path)
            result_image1 = results1[0].plot()
            result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
            Image.fromarray(result_image1[..., ::-1]).save(result_path1)

            # Model 2
            gray_image = cv2.imread(original_image_path)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
            result2 = model_yolov8_2(original_image_path)
            result_image2 = result2[0].plot()
            result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
            Image.fromarray(result_image2[..., ::-1]).save(result_path2)


            return render_template('ObjectDetection.html', image_pred1=result_path1, 
                                   image_pred2=result_path2, image_path=original_image_path)

    return render_template('index.html', image_path=None)
    
# def summarize_results_model(results, model_name):

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

############### live ###############
@app.route('/videos_feed')
def videos_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolov5(frame)
        if results:
            annotated_frame = results[0].plot()
            detected_items = [annotated_frame[0].names[int(box[5])] for box in annotated_frame[0].boxes.data]
            frame_bytes = detected_items.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return jsonify(detected_items=detected_items)

# @app.route('/get_detection_results', methods=['GET'])
# def get_detection_results():
#     global detected_items
#     print("當前檢測到的項目:", detected_items)
#     return jsonify(detected_items=list(set(detected_items)))

def generate_rtsp_stream():
    cap = cv2.VideoCapture(link)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolov5(frame)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def photo1():
    return 'app/static/Screenshot 2024-09-13 003922.png'

@app.route('/live-detect')
def live_detect():
    detected_items = []
    cap = cv2.VideoCapture('rtsp://admin:Abcd1@34@182.239.73.242:8554')
    
    try:
        ret, frame = cap.read()
        if ret:
            results = model_yolov5(frame)
            if results and hasattr(results[0], 'boxes'):
                detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data]
            else:
                logging.error("結果格式不正確或缺少 'boxes'")
        else:
            logging.error("無法從視頻流讀取幀")
    except Exception as e:
        logging.error(f"實時檢測錯誤: {str(e)}")
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
    return Response(live_detect(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)