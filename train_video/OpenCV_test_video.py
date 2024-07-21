import cv2 
import os

cap = cv2.VideoCapture('train_video/thumb.mp4')

while True:  
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('video', frame)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break

# Check if the script file exists
if os.path.exists('train_video/OpenCV_test_video.py'):
    print("成功開啟")
else:
    print("無法開啟")