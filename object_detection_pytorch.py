import cv2
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
from collections import deque
from PIL import Image

# Initialize the YOLOv10 model with the pre-trained weights
yolo_model = YOLO("weights/yolov10n.pt")

# Load the PyTorch model
class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        # Load the VGG16 model architecture
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
        # Replace the classifier with a custom classifier for 2 classes
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 2)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gender_model = GenderModel()
gender_model.load_state_dict(torch.load("best_model.pth"),strict=False)
gender_model = gender_model.to(device)
gender_model.eval()

# Updated COCO class names with "person" and "weapon"
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                  "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush", "weapon"]  # Ensure "weapon" is in your dataset/model

# Gender classes
gender_classes = ['male', 'female']

# Deque to store last N predictions for smoothing
prediction_deque = deque(maxlen=5)

# Function to select the input source
def select_input_source(source_type="video", source_path=""):
    if source_type == "webcam":
        cap = cv2.VideoCapture(0)  # Use the webcam as input
    elif source_type == "rtsp":
        cap = cv2.VideoCapture(source_path)  # Use an RTSP stream as input
    else:
        cap = cv2.VideoCapture(source_path)  # Use a video file as input
    return cap

# Function to stabilize gender prediction
def stabilize_prediction(predictions):
    if len(predictions) > 0:
        return max(set(predictions), key=predictions.count)
    return "unknown"

# Select the input source
source_type = "webcam"  # Change to "webcam" or "rtsp" as needed
source_path = "0"
cap = select_input_source(source_type, source_path)

# Preprocessing transformation for input images
preprocess = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ctime = 0
ptime = 0
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    print(f"Frame Count: {count}")

    # YOLO object detection
    results = yolo_model.predict(frame, conf=0.25, device='cuda:0' if torch.cuda.is_available() else 'cpu')

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            className = cocoClassNames[cls]

            # Only process detections for "person" and "weapon"
            if className in ["person", "knife"]:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop the detected region for gender classification if it's a person
                if className == "person":
                    person_crop = frame[y1:y2, x1:x2]
                    person_crop = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    person_crop = preprocess(person_crop).unsqueeze(0).to(device)

                    # Predict gender
                    with torch.no_grad():
                        gender_conf = gender_model(person_crop)[0]
                        gender_idx = torch.argmax(gender_conf).item()
                        gender_label = gender_classes[gender_idx]

                    # Add prediction to deque
                    prediction_deque.append(gender_label)

                    # Stabilize prediction
                    stabilized_gender = stabilize_prediction(list(prediction_deque))

                    # Display the stabilized gender label
                    label = stabilized_gender
                    color = (255, 0, 0)  # Blue for "person"
                else:
                    # For weapons, use red color and "weapon" label
                    label = className
                    color = (0, 0, 255)  # Red for "weapon"

                # Draw the rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                conf = math.ceil(box.conf[0] * 100) / 100
                label = f"{label}: {conf}"
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # FPS calculation
    ctime = time.time()
    fps = 1 / (ctime - ptime) if ctime != ptime else 0
    ptime = ctime

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(frame, f"Frame Count: {str(count)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()