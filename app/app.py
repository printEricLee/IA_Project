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

# 配置日志输出
logging.basicConfig(level=logging.INFO)

# 初始化 YOLO 模型
model_img = YOLO('model/Iteam_object.pt')  # 图像检测模型
model_wd = YOLO('model/wet_dry.pt')        # 湿/干分类模型
model_test = YOLO('model/yolo11x.pt')      # 测试模型

# 将敏感信息从代码中移除，使用环境变量或配置文件
# 如需使用环境变量，请确保在运行环境中设置以下变量
# link = os.environ.get('RTSP_LINK')
link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'  # 请确保替换为您的实际链接，避免在代码中暴露敏感信息

# 生成唯一文件名
def generate_unique_filename(filename):
    return str(uuid.uuid4()) + os.path.splitext(filename)[1]

# 静态文件处理
@app.route('/index.css')
def serve_static_file():
    return send_from_directory('static', 'index.css')

# 首页
@app.route('/')
def home():
    return render_template('index.html')

# 图片检测页面
@app.route("/objectDetection")
def objectDetection():
    return render_template('ObjectDetection.html')

# 视频检测页面
@app.route("/uploadVideo")
def uploadVideo():
    return render_template('UploadVideo.html')

# 实时检测页面
@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')

@app.route("/loadindpage")
def loadindpage():
    return render_template('LoadingPage.html')

########################################
# 图片检测功能
########################################
@app.route('/imgpred', methods=['GET', 'POST'])
def imgpred():
    if request.method == 'POST':
        # 检查是否有上传图片
        if 'image' not in request.files or request.files['image'].filename == '':
            return redirect(request.url)

        file = request.files['image']
        base_dir = os.path.join('static', 'images')
        os.makedirs(os.path.join(base_dir, 'originals'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'results_model1'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'results_model2'), exist_ok=True)

        # 保存原始图片
        unique_filename = generate_unique_filename(file.filename)
        original_image_path = os.path.join(base_dir, 'originals', unique_filename)
        file.save(original_image_path)

        # 调用通用的处理函数
        result_path1, summary1 = process_image(original_image_path, model_img, base_dir, 'results_model1', 'Model 1')
        result_path2, summary2 = process_image(original_image_path, model_wd, base_dir, 'results_model2', 'Model 2')

        # 将 original_image_path 作为参数传递
        return render_template('ObjectDetection.html', summary1=summary1, image_pred1=result_path1,
                               summary2=summary2, image_pred2=result_path2, image_path=original_image_path)

    return render_template('index.html', image_path=None)

def process_image(image_path, model, base_dir, result_dir_name, model_name):
    # 模型检测
    results = model(image_path)
    result_image = results[0].plot()
    unique_filename = os.path.basename(image_path)
    result_path = os.path.join(base_dir, result_dir_name, f'result_{model_name}_{unique_filename}')
    Image.fromarray(result_image[..., ::-1]).save(result_path)
    summary = summarize_results_model(results, model_name)
    return result_path, summary

@app.route('/image-detect')
def image_detect():
    # 从请求参数中获取 image_path
    original_image_path = request.args.get('image_path')
    detected_items = []

    if original_image_path:
        # 确认模型函数的调用正确
        results = model_img(original_image_path, conf=0.8)
        
        if results and hasattr(results[0], 'boxes'):
            detected_items = [results[0].names[int(box.cls)] for box in results[0].boxes]
        else:
            logging.info("未检测到任何物品")
    else:
        logging.error("未提供有效的图片路径")

    return jsonify(detected_items=detected_items)

# 整理检测结果
def summarize_results_model(results, model_name):
    detected_classes = {}
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = get_class_name(class_id, model_name)
            detected_classes.setdefault(class_name, []).append(confidence)

    summary = [f"{class_name}: {max(scores):.2f}" for class_name, scores in detected_classes.items()]
    return f"{model_name} detected: " + ", ".join(summary) if summary else f"{model_name} detected: No objects detected."

# 获取类别名称（图片）
def get_class_name(class_id, model_name):
    class_map = {
        "Model 1": {0: "construction_waste", 1: "rock", 2: "slurry", 3: "soil"},
        "Model 2": {0: "Dry", 1: "Wet"}
    }
    return class_map.get(model_name, {}).get(class_id, "Unknown")

########################################
# 视频检测功能
########################################
@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('static', 'videos', filename)
    if not os.path.exists(video_path):
        return "Video file not found", 404

    return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    if request.method == 'POST':
        if 'video' not in request.files or request.files['video'].filename == '':
            return redirect(request.url)

        file = request.files['video']
        unique_filename = generate_unique_filename(file.filename)
        video_path = os.path.join('static', 'videos', unique_filename)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        file.save(video_path)

        # 在后台线程中处理视频
        threading.Thread(target=process_video, args=(video_path,)).start()
        return render_template('UploadVideo.html', filename=unique_filename)

    return render_template('UploadVideo.html')

# 处理视频的通用函数
def process_video(video_path):
    logging.info(f"开始处理视频：{video_path}")
    cap = cv2.VideoCapture(video_path)
    output_folder = os.path.join('static', 'processed', str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对每一帧进行检测和标注
        annotated_frame = detect_and_annotate_frame(frame, model_img)

        # 保存标注后的帧
        save_frame(annotated_frame, frame_number, output_folder)
        frame_number += 1

    cap.release()
    # 将处理后的帧合成为视频
    create_video_from_images(output_folder)
    logging.info(f"视频处理完成，结果保存在：{output_folder}")

# 检测并标注帧的通用函数
def detect_and_annotate_frame(frame, model):
    results = model(frame)
    if results and hasattr(results[0], 'boxes'):
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame
    return annotated_frame

# 生成视频流的通用函数
def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"无法打开视频文件：{video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对每一帧进行检测和标注
        annotated_frame = detect_and_annotate_frame(frame, model_img)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# 保存单张帧
def save_frame(frame, frame_number, output_path):
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(filename, frame)

# 从帧创建视频
def create_video_from_images(image_folder):
    output_video_folder = 'static/output_videos'
    os.makedirs(output_video_folder, exist_ok=True)

    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])
    if not images:
        return

    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    video_path = os.path.join(output_video_folder, f'{uuid.uuid4()}.mp4')
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()
    logging.info(f"处理后的视频已保存到：{video_path}")

########################################
# 实时检测功能
########################################

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_stream(link, model_test), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_stream(video_source, model):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logging.error(f"无法连接到视频源：{video_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = detect_and_annotate_frame(frame, model)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/live-detect')
def live_detect():
    detected_items = []
    cap = cv2.VideoCapture(link)

    if not cap.isOpened():
        logging.error(f"无法连接到 RTSP 流：{link}")
        return jsonify(detected_items=detected_items)

    try:
        ret, frame = cap.read()
        if ret:
            results = model_test(frame, conf=0.2)
            if results and hasattr(results[0], 'boxes'):
                detected_items = [results[0].names[int(box.cls)] for box in results[0].boxes]
            else:
                logging.info("未检测到任何物品")
        else:
            logging.error("无法从视频流读取帧")
    except Exception as e:
        logging.error(f"实时检测错误: {str(e)}")
    finally:
        cap.release()

    return jsonify(detected_items=detected_items)

########################################
# 启动应用程序
########################################
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
