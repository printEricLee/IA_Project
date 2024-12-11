from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory, url_for, flash
from ultralytics import YOLO
import cv2
import os
import uuid
import threading
import numpy as np
from PIL import Image
import logging
from flask_mail import Mail, Message

app = Flask(__name__)

# 初始化 YOLO 模型
model_img = YOLO('model/best.pt')  # 圖像檢測模型
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

@app.route("/template")
def template():
    return render_template('template.html')

########################################
# Send Email funtion
########################################
# 配置 Flask-Mail 使用 Gmail SMTP 伺服器
# reference : https://github.com/twtrubiks/Flask-Mail-example?tab=readme-ov-file
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'xyxz55124019@gmail.com'     # Gmail 地址
app.config['MAIL_PASSWORD'] = 'uivh botp tcwb tybz'     # 應用程式密碼
app.secret_key = 'DSANO_1'

mail = Mail(app)

# 發送郵件的輔助函數
def send_email(recipient, subject, body, image_path=None):
    msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient])
    msg.body = body
    # 附加原始圖片
    if image_path:
        with app.open_resource(image_path) as fp:
            msg.attach(os.path.basename(image_path), 'image/jpeg', fp.read())
    mail.send(msg)

# 功能：自動發送郵件
def auto_send_imageResult(summary1, summary2,  image_path):
    recipient = 'xyxz55124019@gmail.com'
    body = "檢測結果:\n"

    try:

        items = ['Slurry', 'dirt', 'nothing', 'other', 'stone']
        for item in items:
            status = '已檢測到' if item in summary1 else '未檢測到'
            body += f"{item}: {status}\n"

        wet_status = '有' if 'wet' in summary2 else '沒有'
        body += f"潮濕情況檢測: {wet_status}\n"

        if 'wet' in summary2:
            body += "警告: 檢測到潮濕！\n"

    finally:
        if all(item not in summary1 for item in ['Slurry', 'dirt', 'nothing', 'other', 'stone']):
            body += f"有沒有 確實也沒有"

    send_email(recipient, "圖片檢測結果", body, image_path)
    flash('功能1結果郵件已自動發送！', 'success')


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

        #Send email
        auto_send_imageResult(summary1, summary2, original_image_path)

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
        "Model 1": {        
            0: "Slurry",
            1: "dirt",
            2: "nothing",
            3: "other",
            4: "stone"},
        "Model 2": {        
            0: "dry",
            1: "wet"}
    }
    return class_map[model_name].get(class_id, "Unknown")
########################################
# 影片檢測功能
########################################
processing = False  # 處理狀態標誌
detected_items = []  # 存儲檢測到的物體列表

def generate_unique_filename(filename):
    return filename  # 生成唯一文件名（當前實現不改變文件名）

def save_frame(frame, frame_number, output_path):
    # 保存幀到指定路徑
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    if cv2.imwrite(filename, frame):
        print(f"成功保存: {filename}")  # 輸出保存成功的消息
    else:
        print(f"無法保存: {filename}")  # 輸出保存失敗的消息

def compress_frame(frame, quality=80):
    # 壓縮幀
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # 設置JPEG質量
    result, buffer = cv2.imencode('.jpg', frame, encode_param)  # 編碼為JPEG
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # 解碼為圖像

def process_video(video_path, output_folder):
    global processing
    logging.info(f"開始處理影片: {video_path}")  # 記錄開始處理影片的日誌

    cap = cv2.VideoCapture(video_path)  # 打開影片文件
    if not cap.isOpened():
        logging.error("錯誤: 無法打開影片文件。")  # 如果無法打開影片，記錄錯誤
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
            logging.info("沒有更多幀可讀或發生錯誤。")  # 如果沒有讀取到幀，記錄信息並退出
            break

        try:
            results = model_img(frame)  # 對幀進行物體檢測
            logging.info(f"幀 {frame_number} 的檢測結果: {results}")  # 記錄檢測結果

            if results:  # 如果有檢測結果
                annotated_frame = results[0].plot()  # 繪製標註幀
                save_frame(annotated_frame, frame_number, output_folder)  # 保存標註幀
                out.write(annotated_frame)  # 將處理後的幀寫入影片文件
                logging.info(f"處理並保存幀 {frame_number}。")  # 記錄保存成功的消息
            else:
                logging.warning("幀中未檢測到任何物體。")  # 如果沒有檢測到物體，記錄警告
                
            frame_number += 1  # 增加幀計數

        except Exception as e:
            logging.error(f"處理幀 {frame_number} 時發生錯誤: {e}")  # 記錄處理過程中的錯誤

    cap.release()  # 釋放影片捕捉對象
    out.release()  # 釋放 VideoWriter
    logging.info("完成影片處理。")  # 記錄處理結束的消息

def create_video_from_images(image_folder):
    output_video_folder = 'static/output_videos'  # 輸出影片的文件夾
    os.makedirs(output_video_folder, exist_ok=True)  # 創建文件夾

    # 獲取文件夾中的所有圖像文件
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    images.sort()  # 按名稱排序圖像

    if not images:
        print("找不到圖片。")  # 如果沒有找到圖像，輸出消息
        return

    first_image = cv2.imread(images[0])  # 讀取第一張圖像
    height, width, _ = first_image.shape  # 獲取圖像的高度和寬度
    video_path = os.path.join(output_video_folder, f'{uuid.uuid4()}.mp4')  # 生成輸出影片的路徑
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))  # 創建影片寫入對象
    print(f"開始寫入影片到: {video_path}")  # 輸出開始寫入影片的消息

    for img_path in images:  # 遍歷所有圖像
        frame = cv2.imread(img_path)  # 讀取圖像
        if frame is not None:
            video_writer.write(frame)  # 寫入圖像幀到影片
            print(f"寫入幀: {img_path}")  # 輸出寫入幀的消息
        else:
            print(f"無法讀取圖片: {img_path}")  # 輸出讀取失敗的消息

    video_writer.release()  # 釋放影片寫入對象
    print("影片寫入完成。")  # 輸出寫入完成的消息

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
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
    cap = cv2.VideoCapture(video_path)  # 打開影片文件
    while cap.isOpened() and processing:  # 當影片仍在打開且正在處理
        ret, frame = cap.read()  # 讀取一幀
        if not ret:
            break  # 如果讀取失敗，則退出

        results = model_img(frame)  # 在幀上運行物體檢測模型
        if results:
            annotated_frame = results[0].plot()  # 繪製標註幀
            compressed_frame = compress_frame(annotated_frame)  # 壓縮幀
            ret, buffer = cv2.imencode('.jpg', compressed_frame)  # 將幀編碼為JPEG格式
            frame_bytes = buffer.tobytes()  # 轉換為字節流

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # 發送幀

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video_frames(os.path.join('static', 'videos', filename)), mimetype='multipart/x-mixed-replace; boundary=frame')  # 返回影片流

@app.route('/videos_feed')
def videos_feed(video_path):
    cap = cv2.VideoCapture(video_path)  # 打開影片文件
    while cap.isOpened() and processing:  # 當影片仍在打開且正在處理
        ret, frame = cap.read()  # 讀取一幀
        if not ret:
            break  # 如果讀取失敗，則退出

        results = model_img(frame)  # 在幀上運行物體檢測模型
        if results:
            annotated_frame = results[0].plot()  # 繪製標註幀
            detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data]  # 獲取檢測到的物體名稱
            compressed_frame = compress_frame(annotated_frame)  # 壓縮幀
            ret, buffer = cv2.imencode('.jpg', compressed_frame)  # 將幀編碼為JPEG格式
            frame_bytes = buffer.tobytes()  # 轉換為字節流

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # 發送幀
    
    return jsonify(detected_items=detected_items)  # 返回檢測到的物體列表

@app.route('/get_detection_results', methods=['GET'])
def get_detection_results():
    global detected_items
    print("當前檢測到的項目:", detected_items)  # 輸出當前檢測到的項目
    return jsonify(detected_items=list(set(detected_items)))  # 返回唯一的檢測項目

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
