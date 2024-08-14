from ultralytics import YOLO
import multiprocessing

def worker():
    print("Worker function")

def predict_file():
    # Load a model
    model = YOLO("runs/detect/train3/weights/best.pt")  # Load a pretrained model

    # Train the model
    model.train(data="stone-dirt-4/data.yaml", conf=0.5, device=0, batch=32)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 可選
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()

    # 在主進程中調用訓練函數
    predict_file()