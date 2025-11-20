from ultralytics import YOLO

MODEL_PATH = "runs/segment/train2/weights/last.pt" 

TEST_SOURCE = "test.jpg"

print(f"\nLoading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


print(f"Running inference on: {TEST_SOURCE}")
model.predict(
    source=TEST_SOURCE,
    save=True,  
    save_txt=True,
    save_conf=True,  
    show=False, 
)

print("\nDone! Check outputs here:")
print("runs/segment/predict/")
print("runs/segment/predict/labels/")
