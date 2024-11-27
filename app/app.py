from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory, url_for
from ultralytics import YOLO
import cv2
import os
import uuid
import threading
import numpy as np
from PIL import Image
import logging

app = Flask(__name__)

# 初始化 YOLO 模型
model_img = YOLO('model/Iteam_object.pt')  # 圖像檢測模型
model_test = YOLO('model/yolo11x.pt')  # 圖像檢測模型
model_wd = YOLO('model/wet_dry.pt')        # 濕/乾分類模型
link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

# 生成唯一檔案名稱
def generate_unique_filename(filename):
    return str(uuid.uuid4()) + os.path.splitext(filename)[1]

# 靜態檔案處理
@app.route('/index.css')
def serve_static_file():
    return send_from_directory('static', 'index.css')

# 首頁
@app.route('/')
def home():
    return render_template('index.html')

# 圖片檢測頁面
@app.route("/objectDetection")
def objectDetection():
    return render_template('ObjectDetection.html')

# 影片檢測頁面
@app.route("/uploadVideo")
def uploadVideo():
    return render_template('UploadVideo.html')

# 即時檢測頁面
@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')

@app.route("/loadindpage")
def loadindpage():
    return render_template('LoadingPage.html')

########################################
# 圖片檢測功能
########################################
@app.route('/imgpred', methods=['GET', 'POST'])
def imgpred():
    if request.method == 'POST':
        # 確認是否有上傳圖片
        if 'image' not in request.files or request.files['image'].filename == '':
            return redirect(request.url)

        file = request.files['image']
        base_dir = os.path.join('static', 'images')
        os.makedirs(os.path.join(base_dir, 'originals'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'results_model1'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'results_model2'), exist_ok=True)

        # 儲存原始圖片
        unique_filename = generate_unique_filename(file.filename)
        original_image_path = os.path.join(base_dir, 'originals', unique_filename)
        file.save(original_image_path)

        # 模型 1 檢測
        results1 = model_img(original_image_path)
        result_image1 = results1[0].plot()
        result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
        Image.fromarray(result_image1[..., ::-1]).save(result_path1)
        summary1 = summarize_results_model(results1, "Model 1")

        # 模型 2 檢測
        results2 = model_wd(original_image_path)
        result_image2 = results2[0].plot()
        result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
        Image.fromarray(result_image2[..., ::-1]).save(result_path2)
        summary2 = summarize_results_model(results2, "Model 2")

        return render_template('ObjectDetection.html', summary1=summary1, image_pred1=result_path1,
                               summary2=summary2, image_pred2=result_path2, image_path=original_image_path)

    return render_template('index.html', image_path=None)

@app.route('/image-detect')
def image_detect():
    # 確保 original_image_path 是可用的
    original_image_path = request.args.get('image_path')  # 假設從請求中獲取圖片路徑
    detected_items = []

    # 確保 model_img 函數的調用正確
    result = model_img(original_image_path, conf=0.8)
    
    if result and hasattr(result[0], 'boxes'):
        detected_items = [result[0].names[int(box[5])] for box in result[0].boxes.data]
    else:
        logging.info("未檢測到任何物品")

    return jsonify(detected_items=detected_items)


# @app.route('/imgpred', methods=['GET', 'POST'])
# def imgpred():
#     if request.method == 'POST':
#         want_json = request.headers.get('Accept') == 'application/json'

#         if 'image' not in request.files or request.files['image'].filename == '':
#             if want_json:
#                 return jsonify({'error': 'No image uploaded'}), 400
#             return redirect(request.url)

#         file = request.files['image']
#         base_dir = os.path.join('static', 'images')
#         os.makedirs(os.path.join(base_dir, 'originals'), exist_ok=True)
#         os.makedirs(os.path.join(base_dir, 'results_model1'), exist_ok=True)
#         os.makedirs(os.path.join(base_dir, 'results_model2'), exist_ok=True)

#         # Save original image
#         unique_filename = generate_unique_filename(file.filename)
#         original_image_path = os.path.join(base_dir, 'originals', unique_filename)
#         file.save(original_image_path)

#         # Model 1 detection
#         results1 = model_img(original_image_path)
#         result_image1 = results1[0].plot()
#         result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
#         Image.fromarray(result_image1[..., ::-1]).save(result_path1)
#         summary1 = summarize_results_model(results1, "Model 1")

#         # Model 2 detection
#         results2 = model_wd(original_image_path)
#         result_image2 = results2[0].plot()
#         result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
#         Image.fromarray(result_image2[..., ::-1]).save(result_path2)
#         summary2 = summarize_results_model(results2, "Model 2")

#         # Prepare response data
#         response_data = {
#             'original_image': original_image_path,
#             'model1': {
#                 'result_image': result_path1,
#                 'summary': summary1
#             },
#             'model2': {
#                 'result_image': result_path2,
#                 'summary': summary2
#             }
#         }

#         if want_json:
#             return jsonify(response_data), 200
#         else:
#             return render_template('ObjectDetection.html', 
#                                 summary1=summary1, 
#                                 image_pred1=result_path1,
#                                 summary2=summary2, 
#                                 image_pred2=result_path2, 
#                                 image_path=original_image_path)

#     return render_template('index.html', image_path=None)

# 整理檢測結果
def summarize_results_model(results, model_name):
    detected_classes = {}
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            class_name = get_class_name(class_id, model_name)
            detected_classes.setdefault(class_name, []).append(confidence)

    summary = [f"{class_name}: {max(scores):.2f}" for class_name, scores in detected_classes.items()]
    return f"{model_name} detected: " + ", ".join(summary) if summary else f"{model_name} detected: No objects detected."

# 取得類別名稱 (圖片)
def get_class_name(class_id, model_name):
    class_map = {
        "Model 1": {0: "construction_waste", 1: "rock", 2: "slurry", 3: "soil"},
        "Model 2": {0: "Dry", 1: "Wet"}
    }
    return class_map[model_name].get(class_id, "Unknown")
########################################
# 影片檢測功能
########################################
@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(process_video(os.path.join('static', 'videos', filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    global processing
    if request.method == 'POST':
        if 'video' not in request.files or request.files['video'].filename == '':
            return redirect(request.url)

        file = request.files['video']
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

# 處理影片
def process_video(video_path, output_folder):
    global processing, detected_items
    print(f"Processing video from: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    detected_items = []
    os.makedirs(output_folder, exist_ok=True)
    frame_number = 0

    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_img(frame)
        if results:
            boxes = results[0].boxes.data
            detected_items.extend([results[0].names[int(box[5])] for box in boxes])
            annotated_frame = results[0].plot()
            compressed_frame = compress_frame(annotated_frame)
            save_frame(compressed_frame, frame_number, output_folder)
            frame_number += 1

            # 確保每一幀都能及時發送
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + compressed_frame + b'\r\n')

    cap.release()
    create_video_from_images(output_folder)

# 儲存單張影格
def save_frame(frame, frame_number, output_path):
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(filename, frame)

# 壓縮影格
def compress_frame(frame, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, buffer = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

# 從影格生成影片
def create_video_from_images(image_folder):
    output_video_folder = 'static/output_videos'
    os.makedirs(output_video_folder, exist_ok=True)

    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    if not images:
        return

    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    video_path = os.path.join(output_video_folder, f'{uuid.uuid4()}.mp4')
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()

########################################
# 即時檢測功能
########################################

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_rtsp_stream():
    cap = cv2.VideoCapture(link)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_img(frame, conf = 0.2)
        detected_items = []

        if results and hasattr(results[0], 'boxes'):
            detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data]

        if 'truck' in detected_items:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/live-detect')
def live_detect():
    detected_items = []
    cap = cv2.VideoCapture(link)

    try:
        ret, frame = cap.read()
        if ret:
            results = model_test(frame, conf = 0.2)
            if results and hasattr(results[0], 'boxes'):
                detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data]
                
                # 檢查是否檢測到卡車
                # if 'truck' not in detected_items:
                #     logging.info("未檢測到卡車，關閉功能或使用其他模型")
                #     # 在這裡可以選擇使用其他模型或關閉功能
                #     return jsonify(message="未檢測到卡車，功能已關閉")
            else:
                logging.info("未檢測到任何物品")
        else:
            logging.error("無法從視頻流讀取幀")
    except Exception as e:
        logging.error(f"實時檢測錯誤: {str(e)}")
    finally:
        cap.release()

    return jsonify(detected_items=detected_items)

########################################
# 啟動應用程式
########################################
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
