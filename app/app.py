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
import os
import logging

# f is the file name 

app = Flask(__name__)

model_yolov8 = YOLO('model/Iteam_object.pt')

model_yolov5 = YOLO('model/For_video.pt')

model_yolov8_2 = YOLO('model/wet_dry.pt')

link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

os.makedirs('runs/detect', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
#os.makedirs('result', exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

############### give every image name ###############
# def generate_unique_filename(filename):
#     _, extension = os.path.splitext(filename)
#     unique_filename = str(uuid.uuid4()) + extension
#     return unique_filename
####################################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')
    
@app.route("/upload")
def image_video_predict():
    return render_template('upload_web.html')

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
            upload_path = os.path.join(basepath, 'uploads', f.filename)
            f.save(upload_path)

            logging.debug(f"File saved to: {upload_path}")

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            result_path = os.path.join(basepath, 'runs/detect', f.filename)

            # Create the directory if it doesn't exist
            result_dir = os.path.dirname(result_path)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            if file_extension in ['jpeg', 'jpg', 'png']:
                img = cv2.imread(upload_path)
                detections = model_yolov8(img, save=True)
                result_image = detections[0].plot()

                if cv2.imwrite(result_path, result_image):
                    image_url = result_path
                    logging.debug(f"Image saved successfully: {result_path}")
                else:
                    logging.error("Failed to save image.")
                    return "Error processing image", 500

            elif file_extension == 'mp4': 
                # Handle video processing here
                pass

            return render_template('upload_web.html', image_url=image_url)

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

@app.route('/live-detect')
def live_detect():
    detected_items = []
    cap = cv2.VideoCapture(link)
    
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
    return Response(generate_rtsp_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)