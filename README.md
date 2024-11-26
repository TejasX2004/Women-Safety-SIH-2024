# Real-Time Gender Classification and Object Detection System

This project uses a combination of YOLO (You Only Look Once) for object detection and a pre-trained gender classification model to analyze video streams in real-time. It also incorporates alerts based on the analysis, such as identifying women alone at night or women surrounded by men.

## Features
- **Object Detection with YOLOv10:** Detects various objects, with a specific focus on identifying persons.
- **Gender Classification:** Uses a pre-trained ResNet-50 model for gender classification of detected people.
- **Alerts:** Sends alerts based on specific conditions like a woman alone at night or women surrounded by men.
- **Real-Time Processing:** Streams are processed in real-time via a WebSocket server.
- **Smoothing:** Gender classification predictions are smoothed using a deque to stabilize results.

## Requirements

### Python Dependencies
- `opencv-python`
- `torch`
- `ultralytics`
- `transformers`
- `PIL`
- `numpy`
- `asyncio`
- `websockets`
- `base64`
- `io`
- `datetime`
- `json`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/repo_name.git
   cd repo_name
   ```
2. Install Dependencies

   ```bash
   pip install -r requirements.txt
   ```
   
## Pre-Trained Models

1. YOLOv10 model weights: Download the yolov10n.pt weights and place them in the weights/ directory.

2. Gender Classification Model: This project uses the pre-trained ResNet-50 model from Hugging Face (microsoft/resnet-50). You don't need to download it manually,    as the transformers library will load it automatically.

## Usage

1.Start the websocket server
```bash
python app.py
```
2. Send data via websocket
    ```json
    {
    "type": "frame",
    "streamId": "unique_stream_id",
    "frame": "base64_encoded_image_data"
    }
   ```

4. Response
  ```json
{
  "alert_type": "alert_type",
  "time": "timestamp",
  "stream_id": "unique_stream_id"
}
```
### Alert types


The system identifies several conditions, including:

1. "Woman alone at night" - If a single woman is detected and the current time is between 8 PM and 4 AM.
2. "Women surrounded by men" - If the number of men exceeds twice the number of women in the frame.
3. "Normal Situation" - For normal Situations.

## Code Overview

1. app.py: The main application script that starts the WebSocket server and processes incoming frames.
2. classify_gender(): Function to classify gender using a pre-trained ResNet-50 model.
3. stabilize_prediction(): Function to stabilize gender predictions over time using a deque.
4. determine_alert_type(): Function to determine the alert type based on the gender and detection data.
5. is_night_time(): Function to check if the current time is between 8 PM and 4 AM.

## Example
![image](https://github.com/user-attachments/assets/324295f1-b941-42ea-ab05-0cf0ffcece06)
### Output:
```json
{
  "alert_type": "Woman surrounded by men",
  "time": "2024-11-25T19:45:00.000000",
  "stream_id": "stream123"
}
```




   

   
