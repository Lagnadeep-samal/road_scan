from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="My Dataset/dataset.yml",
    epochs=20,
    imgsz=640,
    batch=8
)