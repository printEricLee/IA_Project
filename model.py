import cv2
from ultralytics import YOLO
import touch
import os

# data type
accept = ''
unaccepted = ''

# import data(jpg/mp3)
data = []
labels = []

model = touch.load("model_Separate-materials.py")

def load_images(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64,64))
            data.append(img)
            labels.append(label)

load_images(<data name> , <number of data>)