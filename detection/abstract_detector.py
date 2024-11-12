from abc import ABC, abstractmethod
from detection.model_interface import IModel

class Detector(ABC):
    def __init__(self, model: IModel):
        self._model = model

    @abstractmethod
    def detect(self):
        pass

class VideoHandler(ABC):
    @property
    @abstractmethod
    def video_path(self):
        pass

    @video_path.setter
    @abstractmethod
    def video_path(self, value):
        pass

    @property
    @abstractmethod
    def filename(self):
        pass

    @filename.setter
    @abstractmethod
    def filename(self, value):
        pass
