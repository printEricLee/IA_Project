from ultralytics import YOLO
import glob
import multiprocessing

def worker():
    print("Worker function")

def predict_img():
    # Load a model
    model = YOLO("/content/runs/detect/train3/weights/best.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    image_paths = glob.glob('runs/detect/predict/*.jpg')
    results = model(image_paths)  # return a list of Results objects

    # Process results list
    for i, result in enumerate(results):
        # Display the result
        result.show()  # display to screen

        # Save the result with a unique filename
        result.save(filename=f"/content/runs/detect/predict/results/result_{i}.jpg")  # save to disk

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 可選
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()

    # 在主進程中調用訓練函數
    predict_img()