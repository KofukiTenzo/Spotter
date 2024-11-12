# Military Vehicle Detection Web App

This web-based application leverages advanced object detection techniques to automatically identify military vehicles in aerial reconnaissance videos. 
Built using Django, the app employs YOLOv8 models for precise object detection, enabling efficient analysis of surveillance footage.

## Key Features:
- **Video Upload and Processing**: Users can upload video files for analysis.
- **Military Vehicle Detection**: Uses YOLOv8 models to detect military vehicles within the video frames.
- **Ensemble Learning**: Combines the outputs of two YOLOv8 models (small and medium) for enhanced accuracy.
- **Consensus Voting**: Utilizes consensus voting to refine detection results and minimize false positives.
- **Non-Maximum Suppression (NMS)**: Applies NMS to remove duplicate bounding boxes and improve detection reliability.
- **Annotated Video Output**: Processed videos with detected objects (military vehicles) highlighted, available for download.

## Technologies Used:
- **Python**: Core programming language for backend logic.
- **Django**: Framework for the web application.
- **YOLOv8**: Object detection models (small and medium) for detecting military vehicles.
- **PyTorch**: Framework used for deep learning and model inference.
- **OpenCV**: Library for video processing and visualization.
- **Non-Maximum Suppression (NMS)**: To refine bounding box predictions and remove duplicates.

## How it Works:
1. **Upload Video**: The user uploads a video containing aerial reconnaissance footage.
2. **Processing**: The system processes the video using two YOLOv8 models (small and medium) for detecting military vehicles.
3. **Detection & Enhancement**: Ensemble learning and consensus voting improve detection accuracy. NMS filters overlapping bounding boxes.
4. **Results**: The processed video with identified military vehicles is made available for download.
