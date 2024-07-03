import numpy as np
import torch
from pycocotools import mask as mask_util
import torchvision
import os

def get_coco(root, image_set, transforms):
    img_folder = os.path.join(root, "images", image_set)
    ann_file = os.path.join(root, "annotations", f"instances_{image_set}.json")
    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
    return dataset


def get_coco_api_from_dataset(dataset):
    for i in range(len(dataset)):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
    return dataset.coco


def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        mask = mask_util.decode(rle)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    else:
        masks = torch.stack(masks, dim=0)
    return masks


def convert_coco_poly_bbox(segmentations, height, width):
    bboxes = []
    for polygons in segmentations:
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        bbox = mask_util.toBbox(rle)
        bboxes.append(bbox)
    return torch.as_tensor(bboxes, dtype=torch.float32)
