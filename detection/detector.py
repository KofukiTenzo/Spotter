from abc import ABCMeta, abstractmethod
import os

import cv2
import torch
from ultralytics import YOLO

from video_detector import settings
from torchvision.ops import nms

class Detector:
    def __init__(self, video_path, filename, yolov8_path):
        self._video_path = video_path
        self._filename = filename
        self._yolov8_path = yolov8_path
        
    @property
    def video_path(self):
        return self._video_path
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def yolov8_path(self):
        return self._yolov8_path
    
    @video_path.setter  
    def video_path(self, video_path):
        self._video_path = video_path
    
    @filename.setter
    def filename(self, filename):
        self._filename = filename
    
    @yolov8_path.setter
    def yolov8_path(self, yolov8_path):
        self._yolov8_path = yolov8_path
    
    @abstractmethod
    def detect(self):
        pass
        
class Single_Model_Detector(Detector):
    def __init__(self, video_path, filename, yolov8_path):
        # self.set_video_path(video_path)
        # self.set_filename(filename)
        # self.set_yolov8_path(yolov8_path)
        super().__init__(video_path, filename, yolov8_path)
    
    @property
    def video_path(self):
        return self._video_path
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def yolov8_path(self):
        return self._yolov8_path
    
    @video_path.setter  
    def video_path(self, val):
        self._video_path = val
    
    @filename.setter
    def filename(self, val):
        self._filename = val
    
    @yolov8_path.setter
    def yolov8_path(self, val):
        self._yolov8_path = val
        
    def detect(self):
    # Load the YOLOv8 model
        model = YOLO(self.yolov8_path)  # Assuming YOLOv8n model is being used
        # Run detection on each frame
        input_video = os.path.join(settings.MEDIA_ROOT, self.video_path)
        output_video = os.path.join(settings.MEDIA_ROOT, "output", f"detected_{self.filename}")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        
        cap = cv2.VideoCapture(input_video)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = model(frame, augment=True)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


            out.write(frame)

        cap.release()
        out.release()
        return output_video
        
class Ensemble_Model_Detector_Consensus_Voting(Detector):
    def __init__(self, video_path, filename, first_yolov8_path, second_yolov8_path):
        super().__init__(video_path, filename, first_yolov8_path)
        self._second_yolov8_path = second_yolov8_path
        self.__iou_threshold=0.5
        self.__score_threshold=0.3
        
    @property
    def video_path(self):
        return self._video_path
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def first_yolov8_path(self):
        return self._first_yolov8_path
    
    @property
    def second_yolov8_path(self):
        return self._second_yolov8_path
        
    @video_path.setter
    def video_path(self, val):
        self._video_path = val
    
    @filename.setter
    def filename(self, val):
        self._filename = val
        
    @first_yolov8_path.setter
    def first_yolov8_path(self, val):
        self._first_yolov8_path = val
    
    @second_yolov8_path.setter
    def second_yolov8_path(self, val):
        self._second_yolov8_path = val
    
    def detect(self):
        # Load two YOLOv8 models
        model1 = YOLO(self._yolov8_path)  # YOLOv8 small model
        model2 = YOLO(self._second_yolov8_path)  # YOLOv8 medium model

        # Set up video input and output paths
        input_video = os.path.join(settings.MEDIA_ROOT, self._video_path)
        output_video = os.path.join(settings.MEDIA_ROOT, "output", f"detected_{self._filename}")
        cap = cv2.VideoCapture(input_video)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection with both models and augmentation
            results1 = model1(frame, augment=True)
            results2 = model2(frame, augment=True)

            # Gather predictions from both models
            boxes1, scores1, classes1 = self.extract_boxes_scores_classes(results1)
            boxes2, scores2, classes2 = self.extract_boxes_scores_classes(results2)

            # Combine boxes and scores from both models
            combined_boxes = torch.cat((boxes1, boxes2), dim=0)  # Ensure 2D shape
            combined_scores = torch.cat((scores1, scores2))
            combined_classes = torch.cat((classes1, classes2))

            # Consensus voting: Only keep boxes detected by both models
            vouted_boxes, vouted_scores, vouted_classes = self.consensus_voting(combined_boxes, combined_scores, combined_classes, self.__iou_threshold)

            # Apply NMS to the combined predictions
            keep_indices = nms(vouted_boxes, vouted_scores, self.__iou_threshold)
            final_boxes = vouted_boxes[keep_indices]
            final_scores = vouted_scores[keep_indices]
            final_classes = vouted_classes[keep_indices]

            # Draw final boxes on the frame
            for i, box in enumerate(final_boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                score = final_scores[i].item()
                class_id = int(final_classes[i].item())
                if score > self.__iou_threshold:  # Filter based on score threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Write the frame with drawn boxes to the output video
            out.write(frame)

        cap.release()
        out.release()
        return output_video
    
    def extract_boxes_scores_classes(self, results):
        """Helper function to extract bounding boxes, scores, and classes from YOLO results."""
        boxes = []
        scores = []
        classes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])  # Ensure 2D format
                scores.append(box.conf[0].item())
                classes.append(box.cls[0].item())
        return torch.tensor(boxes).reshape(-1, 4), torch.tensor(scores), torch.tensor(classes)

    def consensus_voting(self, boxes, scores, classes, iou_threshold=0.5):
        """Applies consensus voting to keep only the boxes detected by both models."""
        final_boxes, final_scores, final_classes = [], [], []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                # Calculate IoU between each pair of boxes
                iou = self.calculate_iou(boxes[i], boxes[j])
                if iou >= iou_threshold and classes[i] == classes[j]:  # Check if boxes belong to the same class
                    final_boxes.append(boxes[i])
                    final_scores.append(max(scores[i], scores[j]))  # Use max confidence score
                    final_classes.append(classes[i])

        # Check if lists are non-empty before converting to tensors
        if final_boxes:
            return torch.stack(final_boxes), torch.tensor(final_scores), torch.tensor(final_classes)
        else:
            # Return empty tensors if no consensus boxes are found
            return torch.empty((0, 4)), torch.empty(0), torch.empty(0)

    def calculate_iou(self, box1, box2):
        """Helper function to calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0