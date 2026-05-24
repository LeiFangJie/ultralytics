from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.predict(source=0, show=True)  # source=0 表示默认摄像头