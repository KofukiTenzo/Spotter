import os
import cv2
import torch
from ultralytics import YOLO
from django.conf import settings
from torchvision.ops import nms
from abc import ABCMeta, abstractmethod
from django.core.files.storage import FileSystemStorage
from .detector import Ensemble_Model_Detector_Consensus_Voting

class Cleaner():
    @abstractmethod
    def clean_folder():
        pass

class Services():
    def __init__(self) -> None:
        pass
    
    def yolov8_ensemble_detection_on_video_with_consensus_voting(self, video_file):
        self.clean_folder(f'{settings.MEDIA_ROOT}\\output')
        fs = FileSystemStorage()
        
        video_path = fs.save(f"uploaded_videos\\{video_file.name}", video_file)
        filename = video_file.name
        yolo1 = "YOLO\\YOLOv8s.pt"
        yolo2 = "YOLO\\YOLOv8m.pt"
        
        detector = Ensemble_Model_Detector_Consensus_Voting(
            video_path,
            filename,
            yolo1,
            yolo2
        )
        
        result = detector.detect()
        
        detection_result = fs.url(result)
        self.clean_folder(f'{settings.MEDIA_ROOT}\\uploaded_videos')
        
        return detection_result

    def clean_folder(self, directory_path):
        if os.path.exists(directory_path):
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            pass