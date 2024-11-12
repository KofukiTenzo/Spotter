import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from detection.YOLOModel import YOLOModel

from .detector import Ensemble_Model_Detector_Consensus_Voting

class Services():
    def __init__(self) -> None:
        pass
    
    def yolov8_ensemble_detection_on_video_with_consensus_voting(self, video_file):
        self.clean_folder(f'{settings.MEDIA_ROOT}\\output')
        fs = FileSystemStorage()
        
        video_path = fs.save(f"uploaded_videos\\{video_file.name}", video_file)
        filename = video_file.name
        
        model1 = YOLOModel("YOLO\\YOLOv8s.pt")
        model2 = YOLOModel("YOLO\\YOLOv8m.pt")
        
        detector = Ensemble_Model_Detector_Consensus_Voting(
            video_path,
            filename,
            model1,
            model2
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