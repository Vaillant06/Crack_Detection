from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "../crack_detection/runs/segment/train2/weights/best.pt"
STREAM_URL = "http://192.168.225.212:8080/video"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(STREAM_URL)

cv2.namedWindow("Crack Detection", cv2.WINDOW_NORMAL)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.predict(
        frame, 
        conf=0.5, 
        imgsz=640, 
        save=False, 
        show=False
    )

    annotated = results[0].plot()

    cv2.imshow("Crack Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
