from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # base model

model.train(
    data="C:/Users/jatin/Downloads/TACO-master/TACO-master/data.yaml",
    epochs=50,
    imgsz=640
)

model.export(format="onnx")
