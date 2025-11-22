from ultralytics import YOLO

MODEL_PATH = "../crack_detection/runs/segment/train2/weights/best.pt" 

TEST_SOURCE = "test_images/30.jpg"


print(f"\nLoading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


print(f"Running inference on: {TEST_SOURCE}")
model.predict(
    source=TEST_SOURCE,
    save=True,  
    #save_txt=True,
    save_conf=True,  
    show=False,
    # hide_labels=False,
    # hide_conf=False
    # boxes=False 
)

print("\nDone! Check outputs here:")
print("runs/segment/predict/")
