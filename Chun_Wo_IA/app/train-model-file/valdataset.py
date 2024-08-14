from ultralytics import YOLO
import multiprocessing

def worker():
    print("Start working...")

def train_model():
    model = YOLO("runs/detect/train10/weights/best.pt")  

    model.train(data="stone-dirt-5/data.yaml", device=0, batch=13)

if __name__ == '__main__': # for cmd
    multiprocessing.freeze_support()  # select
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()

    train_model()