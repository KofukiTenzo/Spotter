from ultralytics import YOLO
from detection.model_interface import IModel

class YOLOModel(IModel):
    def __init__(self, model_path):
        self._model = YOLO(model_path)

    def detect(self, frame):
        return self._model(frame, augment=True)