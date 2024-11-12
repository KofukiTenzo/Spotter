from abc import ABC, abstractmethod

class IModel(ABC):
    @abstractmethod
    def detect(self, frame):
        pass
    