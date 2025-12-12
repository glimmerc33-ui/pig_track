import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo_models/yolo11n.pt")
    model.train(data="/home/hzq/ultralytics-8.3.163/datasets/data.yaml", 
                cfg="/home/hzq/ultralytics-8.3.163/datasets/default.yaml")