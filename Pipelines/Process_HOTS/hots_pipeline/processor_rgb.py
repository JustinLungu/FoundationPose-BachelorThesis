import shutil
from .base import ModalityProcessor

class RGBProcessor(ModalityProcessor):
    def __init__(self, rgb_file):
        self.rgb_file = rgb_file

    def save_to(self, output_path):
        shutil.copy(self.rgb_file, output_path)
