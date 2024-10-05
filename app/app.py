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
import logging
import torch

app = Flask(__name__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_yolov8 = YOLO('model/Iteam_object.pt')
#model_yolov8.to(device)

model_yolov5 = YOLO('model/For_video.pt')
#model_yolov5.to(device)

model_yolov8_2 = YOLO('model/wet_dry.pt')
#model_yolov8_2.to(device)

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
            
            #original_image_path.to(device)

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
        0: "Slurry",
        1: "dirt",
        2: "nothing",
        3: "other",
        4: "stone"
    }
    
    class_map_model2 = {
        0: "dry",
        1: "wet"
    }

    if model_name == "Model 1":
        return class_map_model1.get(class_id)
    elif model_name == "Model 2":
        return class_map_model2.get(class_id, "unknown")

#====================================================================================================#        
# video
# Global variable to manage processing status

processing = False

def generate_unique_filename(filename):
    return filename

def save_frame(frame, frame_number, output_path):
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    if cv2.imwrite(filename, frame):
        print(f"成功保存: {filename}")
    else:
        print(f"無法保存: {filename}")

def compress_frame(frame, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, buffer = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

def process_video(video_path, output_folder):
    global processing
    video_path.to(device)
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            print("讀取幀失敗，結束處理")
            break

        results = model_yolov5(frame)  # 確保此函數已定義
        if results is None or len(results) == 0:
            print("模型處理失敗，沒有返回結果")
            continue

        # 假設 results[0].plot() 返回標註幀
        annotated_frame = results[0].plot()
        compressed_frame = compress_frame(annotated_frame)  # 壓縮幀
        save_frame(compressed_frame, frame_number, output_folder)
        frame_number += 1

    cap.release()
    create_video_from_images(output_folder)

def create_video_from_images(image_folder):
    output_video_folder = 'static/output_videos'
    os.makedirs(output_video_folder, exist_ok=True)

    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    images.sort()

    if not images:
        print("找不到圖片。")
        return

    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    video_path = os.path.join(output_video_folder, f'{uuid.uuid4()}.mp4')
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    print(f"開始寫入影片到: {video_path}")

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
            print(f"寫入幀: {img_path}")
        else:
            print(f"無法讀取圖片: {img_path}")

    video_writer.release()
    print("影片寫入完成。")

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    global processing
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        file = request.files['video']
        if file and file.filename != '':
            unique_filename = generate_unique_filename(file.filename)
            video_path = os.path.join('static', 'videos', unique_filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            file.save(video_path)

            output_folder = os.path.join('static', 'processed', str(uuid.uuid4()))
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

        results = model_yolov5(frame)
        if results:
            annotated_frame = results[0].plot()
            compressed_frame = compress_frame(annotated_frame)
            ret, buffer = cv2.imencode('.jpg', compressed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video_frames(os.path.join('static', 'videos', filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videos_feed')
def videos_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolov5(frame)
        if results:
            annotated_frame = results[0].plot()
            detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data]
            compressed_frame = compress_frame(annotated_frame)
            ret, buffer = cv2.imencode('.jpg', compressed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return jsonify(detected_items=detected_items)

@app.route('/videos-detect')
def video_detect(video_path):
    detected_items = []
    cap = cv2.VideoCapture(video_path)
    
    try:
        ret, frame = cap.read()
        if ret:
            results = model_yolov5(frame)  # 假設這會返回結果
            # 假設 results[0] 包含檢測框和其他信息
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

#====================================================================================================#

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

def generate_rtsp_stream():
    cap = cv2.VideoCapture(link)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a tensor
        #frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        #frame_tensor = frame_tensor.to(device)

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
    cap = cv2.VideoCapture('rtsp://admin:Abcd1@34@182.239.73.242:8554')
    
    try:
        ret, frame = cap.read()
        if ret:
            results = model_yolov5(frame)  # 假設這會返回結果
            # 假設 results[0] 包含檢測框和其他信息
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

@app.route('/loading-page')
def loadingPage():
    return render_template('loading-page.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)