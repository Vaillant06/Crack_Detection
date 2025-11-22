from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/last.pt")

data_path = "dataset/data.yaml"

model.train(
    data=data_path,
    imgsz=512,  
    epochs=40, 
    batch=2, 
    lr0=0.002,
    device="cpu",  
    workers=1,  
    resume=True
)
