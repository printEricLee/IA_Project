from PIL import Image
import numpy as np
import cv2
import os
# for img and etc
# Load the image using Pillow
img = Image.open('train_pic/preview.jpg')

# Convert to a NumPy array (still in RGB format)
img = np.array(img)

# Convert RGB to BGR
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# change img size
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# Display the image using OpenCV
cv2.imshow('img', img)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# Check if the script file exists
if os.path.exists('train_pic/OpenCV_test_pic.py'):
    print("成功開啟")
else:
    print("無法開啟")