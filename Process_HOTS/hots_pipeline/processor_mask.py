import numpy as np
import cv2
import os
from hots.base import ModalityProcessor

class MaskProcessor(ModalityProcessor):
    def __init__(self, mask_data, label):
        self.mask_data = mask_data
        self.label = label

    def save_to(self, image_name, output_dir):
        binary_mask = (self.mask_data == self.label).astype(np.uint8) * 255
        out_path = os.path.join(output_dir, f"{image_name}.png")
        cv2.imwrite(out_path, binary_mask)
        return out_path
