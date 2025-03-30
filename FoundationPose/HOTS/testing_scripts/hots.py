from typing import *
import os
import cv2
import numpy as np 
from math import ceil
from random import seed, sample

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

SceneTarget = Dict[str, Tensor]
SceneSample = Tuple[Tensor, SceneTarget]

seed(19)


class HOTSScenesDataset(Dataset):
    def __init__(self, root: str, transforms: Callable[[Any], Tensor]):
        self.root = root 
        self.transforms = transforms 
        # Filter the list to keep only PNG files
        self.images = [f for f in sorted(os.listdir(os.path.join(root, "RGB"))) if f.endswith('.png')]
        self.class_masks = sorted(os.listdir(os.path.join(root, "SemanticSegmentation/SegmentationClass")))
        self.inst_masks = sorted(os.listdir(os.path.join(root, "InstanceSegmentation/SegmentationObject")))

    def __getitem__(self, idx: int) -> SceneSample:
        # load images and masks
        img_path = os.path.join(self.root, "RGB", self.images[idx])
        class_mask_path = os.path.join(self.root, "SemanticSegmentation/SegmentationClass", self.class_masks[idx])
        inst_mask_path = os.path.join(self.root, "InstanceSegmentation/SegmentationObject", self.inst_masks[idx])
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        class_mask = np.load(class_mask_path)
        inst_mask = np.load(inst_mask_path)
        
        obj_ids = np.unique(class_mask)[1:]
        class_masks = class_mask == obj_ids[:, None, None]

        obj_ids_1 = np.unique(inst_mask)[1:]
        inst_masks = inst_mask == obj_ids_1[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(class_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0]) 
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = idx
        img_size = img.shape[0:2]

        # convert everything into a torch.Tensor if desired
        if self.transforms is not None:
            H, W = img_size
            img = self.transforms(img)
            boxes = [[xmin/W, ymin/H, xmax/W, ymax/H] for xmin, ymin, xmax, yman in boxes]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            obj_ids = torch.as_tensor(obj_ids, dtype=torch.int64)
            class_masks = torch.as_tensor(class_masks, dtype=torch.uint8)
            inst_masks = torch.as_tensor(inst_masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = torch.as_tensor(area / (H * W))
            img_size = torch.as_tensor([H, W], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = obj_ids
        target["class_masks"] = class_masks
        target["instance_masks"] = inst_masks
        target["image_id"] = image_id
        target["area"] = area
        target["image_size"] = img_size

        return img, target

    def __len__(self):
        return len(self.images)


def load_HOTS_scenes(root='./HOTS_v1', transform=False):
    if isinstance(transform, bool):
        transform = None if not transform else T.ToTensor()
    dataset = HOTSScenesDataset(os.path.join(root, 'scene'), transform)
    # random split
    num_test = ceil(len(dataset) * 0.2)
    indices = sample(list(range(len(dataset))), len(dataset))
    dataset = Subset(dataset, indices[:-num_test])
    dataset_test = Subset(dataset, indices[-num_test:])
    return dataset, dataset_test


def load_HOTS_objects(root='./HOTS_v1/', transform=False):
    if isinstance(transform, bool):
        transform = None if not transform else T.ToTensor()
    dataset = ImageFolder(os.path.join(root, 'object', 'train'), transform=transform)
    dataset_test = ImageFolder(os.path.join(root, 'object', 'test'), transform=transform)
    return dataset, dataset_test