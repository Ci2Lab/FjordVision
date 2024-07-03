import os
import random
import pandas as pd
import torch
import gc
import sys
from PIL import Image
from torchvision import transforms
from anytree.importer import JsonImporter
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools import mask as mask_util
import cv2
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO

sys.path.append('.')

from preprocessing.preprocessing import append_to_parquet_file

importer = JsonImporter()
root = importer.read(open('datasets/ontology.json', 'r'))

classes_file = 'datasets/EMVSD/EMVSD/classes.txt'

species_names = []
with open(classes_file, 'r') as file:
    species_names = [line.strip() for line in file]

genus_names = [node.name for node in root.descendants if node.rank == 'genus']
class_names = [node.name for node in root.descendants if node.rank == 'class']
binary_names = [node.name for node in root.descendants if node.rank == 'binary']

# Path to the model weights
MODEL_PATH = "datasets/MaskRCNN-weights/model_best.pth"
OBJDIR = 'datasets/maskrcnn-segmented-objects/'
CHECKPOINT_FILE = os.path.join(OBJDIR, 'checkpoint.txt')
PARQUET_FILE = 'datasets/maskrcnn-segmented-objects-dataset.parquet'
COCO_ANN_FILE = 'datasets/EMVSD/EMVSD/train_annotations.json'

# Ensure the directory exists
os.makedirs(OBJDIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.3  # Set the confidence threshold

def manage_checkpoint(read=False, update_index=None):
    if read:
        try:
            with open(CHECKPOINT_FILE, 'r') as file:
                return int(file.read())
        except FileNotFoundError:
            return 0  # Return 0 if the file doesn't exist, indicating starting from the beginning
    elif update_index is not None:
        with open(CHECKPOINT_FILE, 'w') as file:
            file.write(str(update_index))

def get_transform(train=False):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)

def apply_mask_to_detected_object(image, bbox, mask, use_mask):
    bbox = bbox.cpu().numpy()
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image[y1:y2, x1:x2]
    if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        return None
    if use_mask and mask is not None:
        mask = mask.squeeze().cpu().numpy()
        mask = mask[y1:y2, x1:x2]
        if mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
            return cropped_img
        mask = (mask * 255).astype(np.uint8)
        if mask.shape[:2] != cropped_img.shape[:2]:
            mask = cv2.resize(mask, (cropped_img.shape[1], cropped_img.shape[0]))
        cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
    return cropped_img

def convert_polygon_to_mask(polygon_points, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 1)
    return mask

def calculate_binary_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:  # Avoid division by zero
        return 0
    return intersection / union

def find_best_ground_truth_match(coco, img_id, predicted_mask_xyn=None, bbox=None, img_shape=None):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    gt_annotations = coco.loadAnns(ann_ids)
    best_iou = 0
    best_class = None
    if predicted_mask_xyn is not None and img_shape is not None:
        predicted_mask = mask_util.decode(predicted_mask_xyn)
        if predicted_mask is None or predicted_mask.size == 0:
            return best_class

        # Threshold the predicted mask
        predicted_mask = (predicted_mask >= 0.5).astype(np.uint8)

        for gt_annotation in gt_annotations:
            gt_mask = coco.annToMask(gt_annotation)
            if gt_mask is None or gt_mask.size == 0:
                continue
            iou = calculate_binary_mask_iou(gt_mask, predicted_mask)
            if iou > best_iou:
                best_iou = iou
                best_class = gt_annotation['category_id']
    elif bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        pred_box_area = (x2 - x1) * (y2 - y1)
        for gt_annotation in gt_annotations:
            gt_mask = coco.annToMask(gt_annotation)
            if gt_mask is None or gt_mask.size == 0:
                continue
            gt_box = cv2.boundingRect(gt_mask)
            gx1, gy1, gx2, gy2 = gt_box
            gt_box_area = (gx2 - gx1) * (gy2 - gy1)
            inter_x1 = max(x1, gx1)
            inter_y1 = max(y1, gy1)
            inter_x2 = min(x2, gx2)
            inter_y2 = min(y2, gy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = pred_box_area + gt_box_area - inter_area
            if union_area == 0:  # Avoid division by zero
                continue
            iou = inter_area / union_area
            if iou > best_iou:
                best_iou = iou
                best_class = gt_annotation['category_id']
    return best_class

def process_result(img_id, img_path, result, basedir, class_index_to_name, coco, max_crops, crop_count):
    data = []
    orig_img = cv2.imread(img_path)
    img_shape = orig_img.shape
    if result['boxes'] is not None:
        for idx, box in enumerate(result['boxes']):
            if crop_count >= max_crops:
                break
            conf = result['scores'][idx].item()
            if conf < CONFIDENCE_THRESHOLD:
                continue  # Skip low-confidence predictions

            cropped_img = apply_mask_to_detected_object(orig_img, box, result['masks'][idx] if result['masks'] is not None else None, True)
            if cropped_img is None:
                continue
            original_filename_stem = Path(img_path).stem
            new_filename = f"{original_filename_stem}_{idx}.jpg"
            path = os.path.join(basedir, new_filename)
            cv2.imwrite(path, cropped_img)
            crop_count += 1
            pred = result['labels'][idx].item()
            species_name = class_index_to_name[int(pred) - 1]
            mask = result['masks'][idx].squeeze().cpu().numpy()
            # Threshold the mask
            mask = (mask >= 0.5).astype(np.uint8)
            predicted_mask_xyn = mask_util.encode(np.asfortranarray(mask))
            best_class = find_best_ground_truth_match(coco, img_id, predicted_mask_xyn=predicted_mask_xyn, img_shape=img_shape)
            if best_class is None:
                continue  # Skip if ground truth is unknown
            true_species_name = class_index_to_name[best_class - 1] if best_class is not None else "Unknown"
            entry = {
                'masked_image': path,
                'confidence': conf,
                'predicted_species': species_name,
                'species': true_species_name
            }
            data.append(entry)
    else:
        print("No detections in this image.")
    return data, crop_count

def process_and_store_batches(image_ids, batch_size, parquet_file_name, max_crops):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(species_names) + 1  # Add 1 for background class
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
    model.to(device)
    model.eval()
    checkpoint_index = manage_checkpoint(read=True)
    total_batches = len(image_ids) // batch_size + (1 if len(image_ids) % batch_size else 0)
    transform = get_transform(train=False)
    coco = COCO(COCO_ANN_FILE)
    crop_count = 0
    for batch_num in range(checkpoint_index, total_batches):
        if crop_count >= max_crops:
            break
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        batch_ids = image_ids[start_index:end_index]
        batch_images = [transform(Image.open(os.path.join('datasets/EMVSD/EMVSD/', coco.imgs[img_id]['file_name'])).convert("RGB")).to(device) for img_id in batch_ids]
        with torch.no_grad():
            batch_results = model(batch_images)
        batch_data = []
        for img_id, result in zip(batch_ids, batch_results):
            if crop_count >= max_crops:
                break
            img_path = os.path.join('datasets/EMVSD/EMVSD/', coco.imgs[img_id]['file_name'])
            entries, crop_count = process_result(img_id, img_path, result, OBJDIR, species_names, coco, max_crops, crop_count)
            batch_data.extend(entries)
        if batch_data:
            df_batch = pd.DataFrame(batch_data)
            append_to_parquet_file(df_batch, parquet_file_name)
        manage_checkpoint(update_index=batch_num + 1)  # Update checkpoint to the next batch
        del batch_data
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        del batch_images
        del batch_results
        torch.cuda.empty_cache()
        gc.collect()

def main():
    coco = COCO(COCO_ANN_FILE)
    img_ids = list(coco.imgs.keys())
    total_images = 5000
    if len(img_ids) < total_images:
        raise ValueError(f"Not enough images in the dataset. Found {len(img_ids)}, but need {total_images}.")
    sampled_img_ids = random.sample(img_ids, total_images)
    process_and_store_batches(sampled_img_ids, 10, PARQUET_FILE, 5000)  # Adjust batch size as needed

if __name__ == "__main__":
    main()