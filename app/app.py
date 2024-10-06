from flask import Flask, render_template, request, redirect, Response, jsonify, send_from_directory 
from ultralytics import YOLO
import cv2
import os
import uuid
import threading
import logging

app = Flask(__name__)

model_yolov8 = YOLO('model/Iteam_object.pt')

model_yolov5 = YOLO('model/For_video.pt')

model_yolov8_2 = YOLO('model/wet_dry.pt')

#====================================================================================================#        
# give all file name
def generate_unique_filename(filename):
    _, extension = os.path.splitext(filename)
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename

#====================================================================================================#        
# set all html route
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

#====================================================================================================#        
# image
@app.route('/imgpred', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or not request.files['image'].filename:
            return redirect(request.url)

        file = request.files['image']
        base_dir = os.path.join('static', 'images')
        os.makedirs(os.path.join(base_dir, 'originals'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'results_model1'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'results_model2'), exist_ok=True)

        unique_filename = generate_unique_filename(file.filename)
        original_image_path = os.path.join(base_dir, 'originals', unique_filename)
        file.save(original_image_path)

        # Model predictions
        results1 = model_yolov8(original_image_path)
        result_image1 = results1[0].plot()
        result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
        cv2.imwrite(result_path1, result_image1)
        summary1 = summarize_results_model(results1, "Model 1")

        result2 = model_yolov8_2(original_image_path)
        result_image2 = result2[0].plot()
        result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
        cv2.imwrite(result_path2, result_image2)
        summary2 = summarize_results_model(result2, "Model 2")

        return render_template('ObjectDetection.html', summary1=summary1, image_pred1=result_path1,
                               summary2=summary2, image_pred2=result_path2, image_path=original_image_path)

    return render_template('index.html', image_path=None)

# boxs name
def summarize_results_model(results, model_name):
    detected_classes = {}

    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            class_name = get_class_name(class_id, model_name)

            detected_classes.setdefault(class_name, []).append(confidence)

    summary = [f"{name}: {max(scores):.2f}" for name, scores in detected_classes.items()]
    return f"{model_name} detected: " + ", ".join(summary) if summary else f"{model_name} detected: No objects detected."

def get_class_name(class_id, model_name):
    class_maps = {
        "Model 1": {0: "Slurry", 1: "dirt", 2: "nothing", 3: "other", 4: "stone"},
        "Model 2": {0: "dry", 1: "wet"}
    }
    return class_maps.get(model_name, {}).get(class_id, "unknown")

#====================================================================================================#        
# video

# send the predicted video to website
@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video_frames(os.path.join('static', 'videos', filename)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def save_frame(frame, frame_number, output_path):
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    if cv2.imwrite(filename, frame):
        print(f"Saved: {filename}")
    else:
        print(f"Failed to save: {filename}")

def compress_frame(frame, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, buffer = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

def process_video(video_path, output_folder):
    global processing
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, ending processing")
            break

        results = model_yolov5(frame)
        if not results:
            print("Model processing failed, no results")
            continue

        annotated_frame = results[0].plot()
        compressed_frame = compress_frame(annotated_frame)
        save_frame(compressed_frame, frame_number, output_folder)
        frame_number += 1

    cap.release()
    create_video_from_images(output_folder)

# use predicted images from video, make its to video
def create_video_from_images(image_folder):
    output_video_folder = 'static/output_videos'
    os.makedirs(output_video_folder, exist_ok=True)

    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    if not images:
        print("No images found.")
        return

    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    video_path = os.path.join(output_video_folder, f'{uuid.uuid4()}.mp4')
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    print(f"Writing video to: {video_path}")

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
            print(f"Wrote frame: {img_path}")
        else:
            print(f"Cannot read image: {img_path}")

    video_writer.release()
    print("Video writing complete.")

# show the predicted video 
@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    global processing
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename:
            video_path = os.path.join('static', 'videos', file.filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            file.save(video_path)

            output_folder = os.path.join('static', 'processed', str(uuid.uuid4()))
            os.makedirs(output_folder, exist_ok=True)

            processing = True
            threading.Thread(target=process_video, args=(video_path, output_folder)).start()
            return render_template('UploadVideo.html', filename=file.filename)

    return render_template('UploadVideo.html')

# predicted video of every frame
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
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# which boxes have predicted from the video
@app.route('/videos-detect', methods=['GET'])
def video_detect():
    video_path = request.args.get('video_path')
    detected_items = {'video': [], 'object': [], 'WetorDry': []}

    cap = cv2.VideoCapture(video_path)
    
    try:
        ret, frame = cap.read()
        if ret:
            for model, key in zip([model_yolov5, model_yolov8, model_yolov8_2], detected_items.keys()):
                results = model(frame)
                if results and hasattr(results[0], 'boxes'):
                    detected_items[key] = [results[0].names[int(box[5])] for box in results[0].boxes.data]
        else:
            return jsonify(error="Unable to read frame from video stream"), 500
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        cap.release()

    return jsonify(detected_items)

#====================================================================================================#        
# live

# send the predicted video to website
@app.route('/rtsp_feed')
def rtsp_feed():
    return Response(generate_rtsp_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# predicted video of every frame
link = 'rtsp://admin:Abcd1@34@182.239.73.242:8554'
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

# which boxes have predicted from the video
@app.route('/live-detect')
def live_detect():
    cap = cv2.VideoCapture('rtsp://admin:Abcd1@34@182.239.73.242:8554')
    detected_items_video, detected_items_object, detected_items_WetorDry = [], [], []
    
    try:
        ret, frame = cap.read()
        if ret:
            models = [model_yolov5, model_yolov8, model_yolov8_2]
            results = [model(frame) for model in models]

            detected_items_video = [results[0][0].names[int(box[5])] for box in results[0][0].boxes.data] if hasattr(results[0][0], 'boxes') else []
            detected_items_object = [results[1][0].names[int(box[5])] for box in results[1][0].boxes.data] if hasattr(results[1][0], 'boxes') else []
            detected_items_WetorDry = [results[2][0].names[int(box[5])] for box in results[2][0].boxes.data] if hasattr(results[2][0], 'boxes') else []
        else:
            logging.error("Unable to read frame from video stream")
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
    finally:
        cap.release()
    
    return jsonify(
        detected_items_video=detected_items_video,
        detected_items_object=detected_items_object,
        detected_items_WetorDry=detected_items_WetorDry
    )

@app.route('/loading-page')
def loadingPage():
    return render_template('loading-page.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9380, debug=True)