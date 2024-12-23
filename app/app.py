from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory, url_for, flash, send_file
# YOLO package
from ultralytics import YOLO
# opencv package
import cv2
import os
# give file name package
import time
from datetime import datetime
import uuid
import threading
import numpy as np
from PIL import Image
import logging
# send email package
from flask_mail import Mail, Message
import random
from io import BytesIO
from flask_cors import CORS
import pandas as pd
# google drive download package
import gdown

def download_model():
    model_file_urls = [
        ('https://drive.google.com/uc?id=1yQElBcqM9uOJC-f33tPB7v9ITfkQh6UC', 'best.pt'),
        ('https://drive.google.com/uc?id=1cnKp-dDsyuXHhpe6wEHJ3gQKAMhHtAki', 'check_truck.pt'),
        ('https://drive.google.com/uc?id=1P3z1DTcbXPVG4hkfdUi1UzOoFj9NPRNZ', 'Iteam_Object.pt'),
        ('https://drive.google.com/uc?id=1pKrpLiHN8IyC7gr9DiYA16rQ-50NWtsB', 'wet_dry.pt')
    ]

    os.makedirs('model', exist_ok=True)

    for model_file_url, model_file_name in model_file_urls:
        model_output_path = os.path.join('model', model_file_name)
        
        if os.path.exists(model_output_path):
            print(f"{model_file_name} is alive!!!")
            continue

        try:
            gdown.download(model_file_url, model_output_path, fuzzy=True)
            if os.path.exists(model_output_path):
                print(f"{model_file_name} ok!!!")
            else:
                print(f"fail of: {model_file_name}")
        except Exception as e:
            print(f"error: {e}")

def download_template_video():
    template_video_file_urls = [
        ('https://drive.google.com/file/d/172ASGowm_Yu2AicpJBhykFZqvghvB7QV/view?usp=sharing', 'Case_1.mp4'),
        ('https://drive.google.com/file/d/1zPyM8yhvGeJbglOqN-y_eHnc91pBd5uI/view?usp=sharing', 'Case_2.mp4'),
        ('https://drive.google.com/file/d/1f7KgiHwewuNxXticIb2DVihAb1IAsmcl/view?usp=sharing', 'Case_3.mp4'),
        ('https://drive.google.com/file/d/14oipJp6_9kwdYHsfgBeNy21Vl2GPJCAB/view?usp=sharing', 'Case_4.mp4'),
        ('https://drive.google.com/file/d/1QeRQHwRoiWC03CVoTmVuwAlrNxBmfjIZ/view?usp=sharing', 'Case_5.mp4'),
        ('https://drive.google.com/file/d/1nFvWEa9cUwC7DAY0eh4rp1r2zFpiql-s/view?usp=sharing', 'Case_6.mp4'),
        ('https://drive.google.com/file/d/1Vutw4l8_WJ_vNpZl93PJFyghgtJdVXR7/view?usp=sharing', 'Case_7.mp4'),
        ('https://drive.google.com/file/d/1_WdBf8iidPZa_QQoTqOIuvlq3dLYNwHo/view?usp=sharing', 'Case_8.mp4')
    ]

    os.makedirs('static/template', exist_ok=True)

    for template_file_url, template_file_name in template_video_file_urls:
        template_video_output_path = os.path.join('static/template', template_file_name)
        
        if os.path.exists(template_video_output_path):
            print(f"{template_file_name} is alive!!!")
            continue

        try:
            gdown.download(template_file_url, template_video_output_path, fuzzy=True)
            if os.path.exists(template_video_output_path):
                print(f"{template_file_name} ok!!!")
            else:
                print(f"fail of: {template_file_name}")
        except Exception as e:
            print(f"error: {e}")

def download_template_image():
    template_image_folder_no_urls = 'https://drive.google.com/drive/folders/1XbcL06-p74vzmH17V376Uj0j7GWWMv3U?usp=sharing'
    template_image_folder_yes_urls = 'https://drive.google.com/drive/folders/1DlDuX0eB95GeCy800XzrHjLKvWQWaGYs?usp=sharing'

    os.makedirs('static/template/no', exist_ok=True)
    os.makedirs('static/template/yes', exist_ok=True)

    template_image_no_output_path = os.path.join('static/template/no')
    template_image_yes_output_path = os.path.join('static/template/yes')

    try:
        gdown.download_folder(template_image_folder_no_urls, template_image_no_output_path)
        gdown.download_folder(template_image_folder_yes_urls, template_image_yes_output_path)
        if os.path.exists(template_image_no_output_path):
            print(f" ok!!!")
        else:
            print(f"fail!!!")
    except Exception as e:
        print(f"error: {e}")

print("start!!!")
download_model()
download_template_video()
download_template_image()
print("finish!!!")

app = Flask(__name__)
CORS(app)

model_img = YOLO('model/Iteam_Object.pt')  # 圖像檢測模型
model_truck = YOLO('model/best.pt')  # 圖像檢測模型
model_wd = YOLO('model/wet_dry.pt')   # 濕/乾分類模型
link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'

# 生成唯一檔案名稱
# def generate_unique_filename(filename):
#     return str(uuid.uuid4()) + os.path.splitext(filename)[1]

# use the date and time for the file name
def generate_unique_filename(filename):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{current_time}{os.path.splitext(filename)[1]}"

@app.route('/model')
def serve_model_file():
    return send_from_directory('model')

# temlate處理
@app.route('/template_folder')
def serve_template_folder():
    return send_from_directory('static', 'template')

# 靜態檔案處理
@app.route('/index.css')
def serve_static_css_file():
    return send_from_directory('static', 'css', 'index.css')

# 靜態檔案處理
@app.route('/script.js')
def serve_static_js_file():
    return send_from_directory('static', 'js', 'script.js')

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

# template(video)
@app.route('/template_video')
def template_video():
    return render_template('template(video).html')

# template(image)
@app.route('/template_image')
def template_image():
    return render_template('template(image).html')

# print('model_truck & detection1:',model_truck.names)
# print('model_img & detection2:',model_img.names)
# print('model_wd & detection3:',model_wd.names)























########################################
# Send Email funtion
########################################

# 配置 Flask-Mail 使用 Gmail SMTP 伺服器
# reference : https://github.com/twtrubiks/Flask-Mail-example?tab=readme-ov-file
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USE_SSL'] = False
# app.config['MAIL_USERNAME'] = 'xyxz55124019@gmail.com'     # Gmail 地址
# app.config['MAIL_PASSWORD'] = 'uivh botp tcwb tybz'     # 應用程式密碼
# app.secret_key = 'DSANO_1'

# mail = Mail(app)

# # 發送郵件的輔助函數
# def send_email(recipient, subject, body, image_path=None):
#     msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient])
#     msg.body = body
#     # 附加原始圖片
#     if image_path:
#         with app.open_resource(image_path) as fp:
#             msg.attach(os.path.basename(image_path), 'image/jpeg', fp.read())
#     mail.send(msg)

# # 功能：自動發送郵件
# def auto_send_imageResult(summary1, summary2,  image_path):
#     recipient = 'xyxz55124019@gmail.com'
#     body = "檢測結果:\n"

#     try:

#         items = ['Slurry', 'dirt', 'nothing', 'other', 'stone']
#         for item in items:
#             status = '已檢測到' if item in summary1 else '未檢測到'
#             body += f"{item}: {status}\n"

#         wet_status = '有' if 'wet' in summary2 else '沒有'
#         body += f"潮濕情況檢測: {wet_status}\n"

#         if 'wet' in summary2:
#             body += "警告: 檢測到潮濕！\n"

#     finally:
#         if all(item not in summary1 for item in ['Slurry', 'dirt', 'nothing', 'other', 'stone']):
#             body += f"有沒有 確實也沒有"

#     send_email(recipient, "圖片檢測結果", body, image_path)
#     flash('功能1結果郵件已自動發送！', 'success')























########################################
# 圖片檢測功能
########################################
results1 = None
results2 = None
@app.route('/imgpred', methods=['GET', 'POST'])
def imgpred():
    if request.method == 'POST':

        global results1
        global results2

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
        result_image1 = Image.fromarray(result_image1)
        result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
        result_image1.save(result_path1)

        summary1 = summarize_results_model(results1, "Model 1") 

        # 模型 2 檢測
        results2 = model_wd(original_image_path)
        result_image2 = results2[0].plot()
        result_image2 = Image.fromarray(result_image2)
        result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
        result_image2.save(result_path2)

        summary2 = summarize_results_model(results2, "Model 2") 

        #Send email
        # auto_send_imageResult(summary1, summary2, original_image_path)

        return render_template('ObjectDetection.html', summary1=summary1, image_pred1=result_path1,
                            summary2=summary2, image_pred2=result_path2, image_path=original_image_path)

    return render_template('index.html', image_path=None)

@app.route('/get_image_results', methods=['GET'])
def get_image_results():
    global results1
    global results2

    detections1 = results1[0].boxes.cls.cpu().numpy().tolist()
    detections2 = results2[0].boxes.cls.cpu().numpy().tolist()

    print("=====")
    print(detections1)
    print(detections2)
    print("=====")

    return jsonify({"Detections1": detections1, "Detections2": detections2})

# 整理檢測結果
def summarize_results_model(results, model_name):
    detected_classes = {}
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            class_name = results[0].names[class_id]
            detected_classes.setdefault(class_name, []).append(confidence)

    summary = [f"{class_name}: {max(scores):.2f}" for class_name, scores in detected_classes.items()]
    return f"{model_name} detected: " + ", ".join(summary) if summary else f"{model_name} detected: No objects detected."























########################################
# 影片檢測功能
########################################
processing = False  # 處理狀態標誌
detected_items = []  # 存儲檢測到的物體列表
detected_items_lock = threading.Lock()  # 鎖以確保線程安全訪問
truck_results = None  # 初始化全局變量
object_results = None  # 初始化全局變量

def save_frame(frame, frame_number, output_path):
    # 保存幀到指定的輸出路徑
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    if cv2.imwrite(filename, frame):
        logging.info(f"成功保存: {filename}")  # 記錄成功
    else:
        logging.error(f"無法保存: {filename}")  # 記錄失敗

def compress_frame(frame, quality=80):
    # 壓縮幀
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # 設置JPEG質量
    result, buffer = cv2.imencode('.jpg', frame, encode_param)  # 編碼為JPEG
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # 解碼為圖像

def process_video(video_path, output_folder):
    # 處理上傳的影片
    global processing
    global detected_items

    logging.info(f"開始處理影片: {video_path}")  # 記錄開始處理影片

    cap = cv2.VideoCapture(video_path)  # 打開影片文件
    if not cap.isOpened():
        logging.error("錯誤: 無法打開影片文件。")  # 記錄錯誤如果無法打開影片
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 獲取幀寬度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 獲取幀高度
    fps = cap.get(cv2.CAP_PROP_FPS)  # 獲取幀率

    # 創建 VideoWriter 來保存處理後的影片
    output_video_path = os.path.join(output_folder, 'output.avi')  # 設置輸出影片路徑
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 設置編碼格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # 初始化 VideoWriter

    frame_number = 0  # 幀計數器

    while cap.isOpened() and processing:  # 當影片仍在打開且正在處理
        ret, frame = cap.read()  # 讀取一幀
        if not ret:
            logging.info("沒有更多幀可讀或發生錯誤。")  # 如果沒有更多幀可讀，記錄信息
            break

        try:
            results = model_img(frame)  # 對幀進行物體檢測
            logging.info(f"幀 {frame_number} 的檢測結果: {results}")  # 記錄檢測結果

            if results:  # 如果有檢測結果
                annotated_frame = results[0].plot()  # 繪製標註幀
                save_frame(annotated_frame, frame_number, output_folder)  # 保存標註幀
                
                # 安全更新檢測到的項目
                with detected_items_lock:
                    detected_items.extend([results[0].names[int(box[5])] for box in results[0].boxes.data])

                out.write(annotated_frame)  # 將處理後的幀寫入影片文件
                logging.info(f"處理並保存幀 {frame_number}。")  # 記錄成功
            else:
                logging.warning("幀中未檢測到任何物體。")  # 如果沒有檢測到物體，記錄警告
                
            frame_number += 1  # 增加幀計數

        except Exception as e:
            logging.error(f"處理幀 {frame_number} 時發生錯誤: {e}")  # 記錄處理過程中的錯誤

    cap.release()  # 釋放影片捕捉對象
    out.release()  # 釋放 VideoWriter
    logging.info("完成影片處理。")  # 記錄處理結束的消息

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    # 處理影片上傳和處理
    global processing
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)  # 如果沒有上傳影片，重定向回原頁面

        file = request.files['video']  # 獲取上傳的文件
        if file and file.filename != '':
            unique_filename = generate_unique_filename(file.filename)  # 生成唯一文件名
            video_path = os.path.join('static', 'videos', unique_filename)  # 設置影片保存路徑
            os.makedirs(os.path.dirname(video_path), exist_ok=True)  # 創建文件夾
            file.save(video_path)  # 保存上傳的影片文件

            output_folder = os.path.join('static', 'processed', str(uuid.uuid4()))  # 創建處理輸出文件夾
            os.makedirs(output_folder, exist_ok=True)

            processing = True  # 設置處理狀態為真
            threading.Thread(target=process_video, args=(video_path, output_folder)).start()  # 啟動新線程處理影片

            return render_template('UploadVideo.html', filename=unique_filename)  # 渲染上傳頁面並傳遞文件名

    return render_template('UploadVideo.html')  # 返回上傳頁面

def generate_video_frames(video_path):
    global truck_results
    global object_results

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")  # 调试信息
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("未能读取到帧")  # 调试信息
            break

        # 第一步：检测卡车
        truck_results = model_truck(frame, conf=0.5, classes=[1,2])
        truck_detected = False
        truck_frame = None
        truck_box = None

        if truck_results and hasattr(truck_results[0], 'boxes'):
            for box in truck_results[0].boxes.data:
                class_id = int(box[5])
                if truck_results[0].names[class_id] == 'box':  # 检测到卡车
                    truck_detected = True
                    # 获取卡车的边界框
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    truck_frame = frame[y1:y2, x1:x2]  # 裁剪卡车区域
                    truck_box = (x1, y1, x2, y2)  # 存储卡车框以便后用
                    break

        detected_items = []
        # 第二步：如果检测到卡车，则在卡车内部进行物体检测
        if truck_detected and truck_frame is not None:
            object_results = model_img(truck_frame)  # 在卡车内部进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制卡车内部物体的边界框
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1_box + x1, y1_box + y1), (x2_box + x1, y2_box + y1), (255, 0, 0), 2)
                    cv2.putText(frame, object_results[0].names[int(box[5])], 
                                (x1_box + x1, y1_box + y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第三步：如果在卡车内未检测到物体，则检测整个画面
        if not detected_items and truck_detected:
            object_results = model_truck(frame, conf=0.5, classes=[1,2])  # 在整个画面进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制卡车外的物体边界框，避免与卡车重叠
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    # 检查边界框是否与卡车重叠
                    if not (truck_box and (truck_box[0] < x2_box and truck_box[2] > x1_box and truck_box[1] < y2_box and truck_box[3] > y1_box)):
                        cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 0), 2)
                        cv2.putText(frame, object_results[0].names[int(box[5])], 
                                    (x1_box, y1_box - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第四步：绘制卡车的边界框（无论是否检测到物体）
        if truck_box:
            cv2.rectangle(frame, (truck_box[0], truck_box[1]), (truck_box[2], truck_box[3]), (0, 255, 0), 2)  # 卡车外框
            cv2.putText(frame, 'Truck', (truck_box[0], truck_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 第五步：如果没有检测到卡车，则进行物体检测
        if not truck_detected:
            object_results = model_truck(frame, conf=0.5, classes=[1,2])  # 在整个画面进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制主画面上的物体边界框
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 0), 2)
                    cv2.putText(frame, object_results[0].names[int(box[5])], 
                                (x1_box, y1_box - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第六步：处理主画面
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 发送主画面和卡车画面（如果可用）
        if truck_frame is not None:
            ret, truck_buffer = cv2.imencode('.jpg', truck_frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                b'--truck-frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + truck_buffer.tobytes() + b'\r\n')
        else:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/get_detection_results', methods=['GET'])
def get_detection_results():
    global truck_results
    global object_results

    if truck_results is None or object_results is None:
        return jsonify({"error": "No detection results available."}), 400

    results1 = truck_results
    results2 = object_results

    detections1 = results1[0].boxes.cls.cpu().numpy().tolist() if results1 else []
    detections2 = results2[0].boxes.cls.cpu().numpy().tolist() if results2 else []

    return jsonify({
        "detections1": detections1,
        "detections2": detections2
    })

@app.route('/video_feed/<filename>')
def video_feed(filename):
    # 串流影片源
    return Response(generate_video_frames(os.path.join('static', 'videos', filename)), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')  # 返回影片流

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify(success=True)




























########################################
# template(video)功能
########################################
@app.route('/template_feed')
def template_feed():
    folder_path = "static/template/"
    video_paths = [file for file in os.listdir(folder_path) if file.endswith(".mp4")]
    
    video_path = os.path.join(folder_path, random.choice(video_paths))
    print("=====")
    print(video_path)
    print("=====")

    return Response(generate_template_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_template_frames(video_path):
    global truck_results
    global object_results

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")  # 调试信息
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("未能读取到帧")  # 调试信息
            break

        # 第一步：检测卡车
        truck_results = model_truck(frame, conf=0.5, classes=[1,2])
        truck_detected = False
        truck_frame = None
        truck_box = None

        if truck_results and hasattr(truck_results[0], 'boxes'):
            for box in truck_results[0].boxes.data:
                class_id = int(box[5])
                if truck_results[0].names[class_id] == 'box':  # 检测到卡车
                    truck_detected = True
                    # 获取卡车的边界框
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    truck_frame = frame[y1:y2, x1:x2]  # 裁剪卡车区域
                    truck_box = (x1, y1, x2, y2)  # 存储卡车框以便后用
                    break

        detected_items = []
        # 第二步：如果检测到卡车，则在卡车内部进行物体检测
        if truck_detected and truck_frame is not None:
            object_results = model_img(truck_frame)  # 在卡车内部进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制卡车内部物体的边界框
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1_box + x1, y1_box + y1), (x2_box + x1, y2_box + y1), (255, 0, 0), 2)
                    cv2.putText(frame, object_results[0].names[int(box[5])], 
                                (x1_box + x1, y1_box + y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第三步：如果在卡车内未检测到物体，则检测整个画面
        if not detected_items and truck_detected:
            object_results = model_truck(frame, conf=0.5, classes=[1,2])  # 在整个画面进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制卡车外的物体边界框，避免与卡车重叠
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    # 检查边界框是否与卡车重叠
                    if not (truck_box and (truck_box[0] < x2_box and truck_box[2] > x1_box and truck_box[1] < y2_box and truck_box[3] > y1_box)):
                        cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 0), 2)
                        cv2.putText(frame, object_results[0].names[int(box[5])], 
                                    (x1_box, y1_box - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第四步：绘制卡车的边界框（无论是否检测到物体）
        if truck_box:
            cv2.rectangle(frame, (truck_box[0], truck_box[1]), (truck_box[2], truck_box[3]), (0, 255, 0), 2)  # 卡车外框
            cv2.putText(frame, 'Truck', (truck_box[0], truck_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 第五步：如果没有检测到卡车，则进行物体检测
        if not truck_detected:
            object_results = model_truck(frame, conf=0.5, classes=[1,2])  # 在整个画面进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制主画面上的物体边界框
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 0), 2)
                    cv2.putText(frame, object_results[0].names[int(box[5])], 
                                (x1_box, y1_box - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第六步：处理主画面
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 发送主画面和卡车画面（如果可用）
        if truck_frame is not None:
            ret, truck_buffer = cv2.imencode('.jpg', truck_frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                b'--truck-frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + truck_buffer.tobytes() + b'\r\n')
        else:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/template_video_info')
def template_video_info():

    global truck_results
    global object_results

    results1 = truck_results
    results2 = object_results

    detections1 = []
    detections2 = []
    
    detections1 = results1[0].boxes.cls.cpu().numpy().tolist()
    detections2 = results2[0].boxes.cls.cpu().numpy().tolist()

    return jsonify({
        "detections1": detections1,
        "detections2": detections2
    })
    



























########################################
# template(image)功能
########################################
@app.route('/template_image_feed')
def template_image_feed(): 

    global image_path

    folder_path_yes = "static/template/yes"
    folder_path_no = "static/template/no"

    image_paths_yes = [file for file in os.listdir(folder_path_yes) if file.endswith(('.jpg', '.jpeg'))]
    image_paths_no = [file for file in os.listdir(folder_path_no) if file.endswith(('.jpg', '.jpeg'))]

    image_paths = image_paths_yes + image_paths_no
    
    image_path = os.path.join(random.choice([folder_path_yes, folder_path_no]), random.choice(image_paths))

    print("=====")
    print(image_path)
    print("=====")

    image = cv2.imread(image_path)
    result = model_img(image)

    result_image = result[0].plot()

    _, buffer = cv2.imencode('.jpg', result_image)
    img_io = BytesIO(buffer)

    return send_file(img_io, mimetype='image/jpeg')

@app.route('/template_image_info')
def template_image_info():
    global image_path 

    image = cv2.imread(image_path)
    im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    im_gray_mat = Image.fromarray(im_gray)

    result1 = model_img(image)
    result2 = model_wd(im_gray_mat)

    box1 = result1[0].boxes
    box2 = result2[0].boxes

    detections1 = box1.cls.cpu().numpy().tolist()
    detections2 = box2.cls.cpu().numpy().tolist()

    print("=====")
    print(detections1)
    print(detections2)
    print("=====")
    print(model_img.names)
    print(model_wd.names)
    print("=====")

    return jsonify({"detections1": detections1, "detections2": detections2})


















########################################
# 即時檢測功能
########################################
frame_obj = None
frame_wd = None

@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_rtsp_stream():
#     cap = cv2.VideoCapture(link)

#     global frame_obj
#     global frame_wd

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_obj = model_img(frame, conf = 0.6)
#         frame_wd = model_wd(frame, conf = 0.6)

#         annotated_frame = frame_obj[0].plot()
#         ret, buffer = cv2.imencode('.jpg', annotated_frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()
#     cv2.destroyAllWindows()

def generate_rtsp_stream():
    global truck_results
    global object_results

    cap = cv2.VideoCapture(link)
    if not cap.isOpened():
        print("无法打开视频文件")  # 调试信息
        return

    while True:
        ret, frame = cap.read()  # 读取视频帧
        if not ret:
            print("无法读取视频帧")
            break

        # 第一步：检测卡车
        truck_results = model_truck(frame, conf=0.5, classes=[1, 2])
        truck_detected = False
        truck_frame = None
        truck_box = None

        if truck_results and hasattr(truck_results[0], 'boxes'):
            for box in truck_results[0].boxes.data:
                class_id = int(box[5])
                if truck_results[0].names[class_id] == 'box':  # 检测到卡车
                    truck_detected = True
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    truck_frame = frame[y1:y2, x1:x2]  # 裁剪卡车区域
                    truck_box = (x1, y1, x2, y2)  # 存储卡车框以便后用
                    break

        detected_items = []
        # 第二步：如果检测到卡车，则在卡车内部进行物体检测
        if truck_detected and truck_frame is not None:
            object_results = model_img(truck_frame)  # 在卡车内部进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制卡车内部物体的边界框
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1_box + x1, y1_box + y1), (x2_box + x1, y2_box + y1), (255, 0, 0), 2)
                    cv2.putText(frame, object_results[0].names[int(box[5])], 
                                (x1_box + x1, y1_box + y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第三步：如果在卡车内未检测到物体，则检测整个画面
        if not detected_items and truck_detected:
            object_results = model_truck(frame, conf=0.5, classes=[1, 2])  # 在整个画面进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制卡车外的物体边界框，避免与卡车重叠
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if not (truck_box and (truck_box[0] < x2_box and truck_box[2] > x1_box and truck_box[1] < y2_box and truck_box[3] > y1_box)):
                        cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 0), 2)
                        cv2.putText(frame, object_results[0].names[int(box[5])], 
                                    (x1_box, y1_box - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第四步：绘制卡车的边界框（无论是否检测到物体）
        if truck_box:
            cv2.rectangle(frame, (truck_box[0], truck_box[1]), (truck_box[2], truck_box[3]), (0, 255, 0), 2)  # 卡车外框
            cv2.putText(frame, 'Truck', (truck_box[0], truck_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 第五步：如果没有检测到卡车，则进行物体检测
        if not truck_detected:
            object_results = model_truck(frame, conf=0.5, classes=[1, 2])  # 在整个画面进行物体检测
            if object_results and hasattr(object_results[0], 'boxes'):
                detected_items = [object_results[0].names[int(box[5])] for box in object_results[0].boxes.data]

                # 绘制主画面上的物体边界框
                for box in object_results[0].boxes.data:
                    x1_box, y1_box, x2_box, y2_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 0), 2)
                    cv2.putText(frame, object_results[0].names[int(box[5])], 
                                (x1_box, y1_box - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # 第六步：处理主画面
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 发送主画面和卡车画面（如果可用）
        if truck_frame is not None:
            ret, truck_buffer = cv2.imencode('.jpg', truck_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n' +
                   b'--truck-frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + truck_buffer.tobytes() + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()  # 释放视频捕捉对象


@app.route('/get_rtsp_results', methods=['GET'])
def get_rtsp_results():

    global frame_obj
    global frame_wd

    detections1 = frame_obj[0].boxes.cls.cpu().numpy().tolist()
    detections2 = frame_wd[0].boxes.cls.cpu().numpy().tolist()

    print("=====")
    print(detections1)
    print(detections2)
    print("=====")
    print(model_img.names)
    print(model_wd.names)
    print("=====")

    return jsonify({"detections1": detections1, "detections2": detections2})

########################################
# 啟動應用程式
########################################
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

