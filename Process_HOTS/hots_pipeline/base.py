from abc import ABC, abstractmethod

class ModalityProcessor(ABC):
    @abstractmethod
    def save_to(self):
        pass
