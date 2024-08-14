from ultralytics import YOLO
import multiprocessing

def worker(): # for cmd
    print("Start working")

def train_model():
    # Load model ('yolov8m.pt')
    model = YOLO('yolov9e.pt')
    model.half()  # Load a pretrained model

    # Train the model
    model.train(data="object--2/data.yaml", epochs=10, device=0, batch=15)

if __name__ == '__main__': # for cmd
    multiprocessing.freeze_support()  # select 
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()

    train_model()