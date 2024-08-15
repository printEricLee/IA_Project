from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory, send_file, session
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import uuid
import threading
import time
import numpy as np

app = Flask(__name__)

# use model 1 ( predict what is it )
model1 = YOLO('best1_v4.pt')

# use model 2 ( predict the state )
model2 = YOLO('best2_v2.pt')


#====================================================================================#


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
    

#====================================================================================#

@app.route('/imgpred', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查是否上传文件
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        
        # 检查用户是否选择了文件
        if file.filename == '':
            return redirect(request.url)

        if file:
            unique_filename = generate_unique_filename(file.filename)
            image_path = os.path.join('static', 'images', unique_filename)
            file.save(image_path)
            
            try:
            # Model predictions
                results1 = model1(image_path)
                result_image1 = results1[0].plot()  # 假设只处理第一个结果
                result_path1 = os.path.join('static', 'images', 'result_model1_' + unique_filename)
                Image.fromarray(result_image1[..., ::-1]).save(result_path1)
                
                summary1 = summarize_results_model(results1, "Model 1")

                results2 = model2(image_path)
                result_image2 = results2[0].plot()  # 假设只处理第一个结果
                result_path2 = os.path.join('static', 'images', 'result_model2_' + unique_filename)
                Image.fromarray(result_image2[..., ::-1]).save(result_path2)
                
                summary2 = summarize_results_model(results2, "Model 2")
                
                # Check if "wet" is detected
                if "wet" in summary2:
                    alert_message = "Warning: Wet condition detected!"
                else:
                    alert_message = None

        
                return render_template('ObjectDetection.html', summary1=summary1, summary2=summary2, image_pred1=result_path1, image_pred2=result_path2, image_path=image_path, alert_message=alert_message)

            except Exception as e:
                return render_template('ObjectDetection.html', error=f"发生错误: {e}")

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
    # Map class IDs to names based on the model
    class_map_model1 = {
        0: "dirt",
        1: "stone",
        # Add other classes for model 1
    }
    
    class_map_model2 = {
        0: "dry",
        1: "nothing",
        2: "uk",
        3: "wet",
        # Add other classes for model 2
    }

    if model_name == "Model 1":
        return class_map_model1.get(class_id, "unknown")
    elif model_name == "Model 2":
        return class_map_model2.get(class_id, "unknown")
        
        
#====================================================================================#
@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
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

            # Start processing in a background thread
            threading.Thread(target=process_video, args=(video_path,)).start()

            return redirect(url_for('vidpred'))  # Redirect to show results

    # Display results if available
    summary1 = session.get('summary1', None)
    summary2 = session.get('summary2', None)
    
    return render_template('UploadVideo.html', summary1=summary1, summary2=summary2)

def process_video(video_path):
    # Process the video with models
    results1 = model1(video_path)
    results2 = model2(video_path)

    # Summarize results
    summary1 = summarize_results_model(results1, "Model 1")
    summary2 = summarize_results_model(results2, "Model 2")
    
    # Store results in session
    session['summary1'] = summary1
    session['summary2'] = summary2
#====================================================================================#



@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_frames():
    cap = cv2.VideoCapture(0)  # Use the default webcam

    while True:
        success, frame = cap.read()

        if success:
            # Perform prediction with model1
            results1 = model1(frame)
            annotated_frame1 = results1[0].plot()

            # Perform prediction with model2
            results2 = model2(frame)
            annotated_frame2 = results2[0].plot()

            # Combine the annotated frames from both models
            combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame2, 0.5, 0)

            # Convert the combined frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', combined_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame bytes as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    cap.release()

def delete_images_after_delay():
    while True:
        time.sleep(86400)  # Wait 1 day
        image_folder = 'static/images'
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Flask route to delete images after 2 minutes

#====================================================================================#

@app.route('/delete', methods=['GET'])
def delete():
    threading.Thread(target=delete_images_after_delay).start()
    return jsonify({"message": "Images will be deleted continuously after 2 minutes."})


 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)