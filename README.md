### Crack Detection using YOLOv8 Segmentation

This project implements a crack-detection system using YOLOv8 Segmentation, trained to identify and outline cracks on surfaces such as roads, pavements, walls, and other concrete structures.
The model outputs segmentation masks that follow the crack exactly — not just bounding boxes.


## Features

YOLOv8s-seg segmentation model

Pixel-level crack boundary detection

Resume-training support

Prediction with confidence scores

CPU-friendly training configuration

Clean dataset format (train / valid / test)


## Project Structure

Crack_Detection/
│
├── train_model.py
├── test_model.py
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md   ← this file


## Dataset Folder

-- Dataset must follow this exact structure

dataset/
│── train/
│   ├── images/
│   └── labels/
│── valid/
│   ├── images/
│   └── labels/
│── test/
    ├── images/
    └── labels/


### Installation

# 1. Clone your repository
git clone https://github.com/<your-username>/crack_detection_yolov8.git
cd crack_detection_yolov8/Crack_Detection


# 2. Install dependencies
pip install -r requirements.txt


# 3. Start Training
python train_model.py


# 4. Training outputs go to:
runs/segment/train*/


# 5. Update train_model.py to load the last checkpoint:
model = YOLO("runs/segment/train2/weights/last.pt")
# Or resume from the best model:
model = YOLO("runs/segment/train2/weights/best.pt")


# 6. Then run:
python train_model.py


# 7. To test the model:
python test_model.py


# 8. Prediction images will be saved here:
runs/segment/predict/


### You will get:
1.segmented crack image
2.confidence scores
3.mask overlay


### Built With
Python 3.x
Ultralytics YOLOv8
OpenCV
Roboflow


### License
This project is intended for educational and research purposes.
Feel free to modify, extend, and build upon it.


### Acknowledgements
Dataset processed using Roboflow
YOLOv8 by Ultralytics