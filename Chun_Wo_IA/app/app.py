from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory 
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import uuid
import threading
import time
import numpy as np
import subprocess

app = Flask(__name__)

model_yolov8 = YOLO('model/Iteam_object.pt')

model_yolov5 = YOLO('model/For_video.pt')

model_yolov8_2 = YOLO('model/wet_dry.pt')

# give every image name
def generate_unique_filename(filename):
    _, extension = os.path.splitext(filename)
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')
    
@app.route("/uploadVideo")
def uploadVideo():
    return render_template('UploadVideo.html')
    
@app.route("/objectDetection")
def objectDetection():
    return render_template('ObjectDetection.html')
    
@app.route('/index.css')
def serve_static_file():
    return send_from_directory('static', 'index.css')

@app.route('/imgpred', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload image
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        
        # Check user uploaded image
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Ensure the required directories exist
            base_dir = os.path.join('static', 'images')
            os.makedirs(os.path.join(base_dir, 'originals'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'results_model1'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'results_model2'), exist_ok=True)

            unique_filename = generate_unique_filename(file.filename)
            original_image_path = os.path.join(base_dir, 'originals', unique_filename)
            file.save(original_image_path)
            
            # Model predictions
            # Model 1
            results1 = model_yolov8(original_image_path)
            result_image1 = results1[0].plot()
            result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
            Image.fromarray(result_image1[..., ::-1]).save(result_path1)
            summary1 = summarize_results_model(results1, "Model 1")

            # Model 2
            gray_image = cv2.imread(original_image_path)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
            result2 = model_yolov8_2(original_image_path)
            result_image2 = result2[0].plot()
            result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
            Image.fromarray(result_image2[..., ::-1]).save(result_path2)
            summary2 = summarize_results_model(result2, "Model 2")

            return render_template('ObjectDetection.html', summary1=summary1, image_pred1=result_path1, summary2=summary2, 
                                   image_pred2=result_path2, image_path=original_image_path)

    return render_template('index.html', image_path=None)
    
def summarize_results_model(results, model_name):
    detected_classes = {}
    
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            class_name = get_class_name(class_id, model_name)  # Pass model_name for differentiation
            
            if class_name in detected_classes:
                detected_classes[class_name].append(confidence)
            else:
                detected_classes[class_name] = [confidence]

    summary = []
    for class_name, scores in detected_classes.items():
        max_confidence = max(scores)
        summary.append(f"{class_name}: {max_confidence:.2f}")

    return f"{model_name} detected: " + ", ".join(summary) if summary else f"{model_name} detected: No objects detected."

def get_class_name(class_id, model_name):
    class_map_model1 = {
        0: "dirt",
        1: "stone",
    }
    
    class_map_model2 = {
        0: "dry",
        1: "wet"
    }

    if model_name == "Model 1":
        return class_map_model1.get(class_id, "unknown")
    elif model_name == "Model 2":
        return class_map_model2.get(class_id, "unknown")
        
# video
# Global variable to manage processing status
processing = False

def generate_unique_filename(filename):
    return filename

def save_frame(frame, frame_number, output_path):
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(filename, frame)

def process_video(video_path, output_folder):
    global processing
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            break

        results3 = model_yolov5(frame)
        annotated_frame3 = results3[0].plot()

        save_frame(annotated_frame3, frame_number, output_folder)
        frame_number += 1

    cap.release()

    # Get the folder name for naming the output video
    folder_name = os.path.basename(os.path.normpath(output_folder))
    output_video_folder = os.path.join('static', 'output_videos')  # Specify your desired output folder
    os.makedirs(output_video_folder, exist_ok=True)

    output_video_path = os.path.join(output_video_folder, f'{folder_name}_output_video.mp4')
    create_video_from_images(output_folder, output_video_path)
    print(f"Video created at: {output_video_path}")

def create_video_from_images(image_folder, output_video_path, fps=30):
    images = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith((".jpg", ".png")):
                images.append(os.path.join(root, file))

    images.sort()
    print(f"Found images: {images}")

    if not images:
        print("No images found in the directory.")
        return

    first_image_path = images[0]
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Failed to read the first image: {first_image_path}")
        return

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue
        video_writer.write(frame)
        print(f"Written frame: {img_path}")

    video_writer.release()
    print(f"Video created at {output_video_path}")

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    global processing
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        file = request.files['video']
        
        if file.filename == '':
            return redirect(request.url)

        if file:
            unique_filename = generate_unique_filename(file.filename)
            video_path = os.path.join('static', 'videos', unique_filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            file.save(video_path)

            # Create a unique folder for processed frames using a UUID
            output_folder_name = str(uuid.uuid4())
            output_folder = os.path.join('static', 'processed', output_folder_name)
            os.makedirs(output_folder, exist_ok=True)
            processing = True
            threading.Thread(target=process_video, args=(video_path, output_folder)).start()

            return render_template('UploadVideo.html', filename=unique_filename)

    return render_template('UploadVideo.html')

def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            break

        results3 = model_yolov5(frame)
        annotated_frame3 = results3[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame3)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video_frames(os.path.join('static', 'videos', filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

processing = True

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify(success=True)

def process_video():
    while processing:
        # 这里是您的视频处理代码
        pass

# 存储检测到的物体及其时间
detections = []


@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

path = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

def generate_rtsp_stream():
    cap = cv2.VideoCapture(path)

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)