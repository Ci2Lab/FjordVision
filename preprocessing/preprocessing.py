import numpy as np
import cv2
import numpy as np
import io
from pathlib import Path
import pandas as pd
import os
from anytree import find_by_attr
from anytree import RenderTree


def apply_mask_to_detected_object(orig_img, box, mask, use_masks):
    """
    Crop the detected object in an image specified by the bounding box
    and optionally apply the mask to the detected object.

    :param orig_img: The original image as a NumPy array.
    :param box: The bounding box object with 'xyxy' attribute.
    :param mask: The mask object with 'xy' attribute for segments.
    :param use_masks: Boolean indicating whether to use masks or not.
    :return: Cropped image of the detected object.
    """
    # Get bounding box (bbox) coordinates
    bbox = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, bbox)

    if x1 < 0 or y1 < 0 or x2 > orig_img.shape[1] or y2 > orig_img.shape[0]:
        print(f"Skipping invalid bbox: {bbox}")
        return None

    # Crop the image using bbox
    cropped_img_np = orig_img[y1:y2, x1:x2]

    if cropped_img_np.size == 0:
        print(f"Skipping empty crop for bbox: {bbox}")
        return None

    if use_masks and mask is not None:
        # Initialize an empty binary mask with the same dimensions as the original image
        npmask = np.zeros(orig_img.shape[:2], dtype=np.uint8)

        # Fill the mask using the segments in 'xy'
        for segment in mask.xy:
            np_segment = np.array(segment, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(npmask, [np_segment], 255)

        # Crop and resize the mask to match the cropped image size
        cropped_mask = npmask[y1:y2, x1:x2]
        mask_resized = cv2.resize(cropped_mask, (x2 - x1, y2 - y1))

        # Apply the mask to the cropped image
        masked_image = cv2.bitwise_and(cropped_img_np, cropped_img_np, mask=mask_resized)
        return masked_image
    else:
        return cropped_img_np

def calculate_binary_mask_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) for binary masks.

    :param mask1: First binary mask as a NumPy array.
    :param mask2: Second binary mask as a NumPy array.
    :return: IoU score as a float.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0  # To handle cases where both masks are empty
    return np.sum(intersection) / np.sum(union)


def convert_polygon_to_mask(polygon, img_shape):
    """
    Convert polygon coordinates to a binary mask.

    :param polygon: Polygon coordinates as a list of (x, y) tuples.
    :param img_shape: Shape of the corresponding image (height, width).
    :return: Binary mask as a NumPy array.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)  # Create a blank mask
    if len(polygon) > 0:  # Check if the polygon has points
        np_polygon = np.array(polygon) * [img_shape[1], img_shape[0]]  # Scale to image size
        np_polygon = np_polygon.astype(np.int32)  # Convert to integer
        cv2.fillPoly(mask, [np_polygon], 255)  # Fill the polygon on the mask
    return mask


def load_ground_truth_mask_xyn(label_file):
    """
    Load ground truth mask data from a label file and return in mask.xyn format.

    :param label_file: Path to the label file.
    :return: List of tuples (class_index, mask_xyn).
    """
    gt_masks_xyn = []
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found: {label_file}")
        return gt_masks_xyn  # Return an empty list if the file doesn't exist

    with open(label_file, 'r') as file:
        for line in file:
            elements = line.strip().split()
            class_index = int(elements[0])
            mask_xyn = np.array([float(x) for x in elements[1:]]).reshape((-1, 2))
            gt_masks_xyn.append((class_index, mask_xyn))

    return gt_masks_xyn


def find_best_ground_truth_match(result, predicted_mask_xyn=None, img_shape=None, bbox=None):
    """
    Find the ground truth annotation that best matches the predicted mask or bounding box.

    :param result: The result object containing the image path and masks or bounding boxes.
    :param predicted_mask_xyn: The predicted mask in xyn format (optional).
    :param img_shape: Shape of the corresponding image (optional).
    :param bbox: The predicted bounding box (optional).
    :return: The best matching ground truth class.
    """
    label_file = result.path.replace('/images/', '/labels/').replace('.jpg', '.txt')
    gt_annotations = load_ground_truth_mask_xyn(label_file)

    best_iou = 0
    best_class = None

    if predicted_mask_xyn is not None and img_shape is not None:
        predicted_mask = convert_polygon_to_mask(predicted_mask_xyn, img_shape)

        for gt_annotation in gt_annotations:
            cls, gt_polygon_points = gt_annotation
            gt_mask = convert_polygon_to_mask(gt_polygon_points, img_shape)
            iou = calculate_binary_mask_iou(gt_mask, predicted_mask)

            if iou > best_iou:
                best_iou = iou
                best_class = cls
    elif bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        pred_box_area = (x2 - x1) * (y2 - y1)

        for gt_annotation in gt_annotations:
            cls, gt_polygon_points = gt_annotation
            gt_mask = convert_polygon_to_mask(gt_polygon_points, img_shape)
            gt_box = cv2.boundingRect(gt_mask)
            gx1, gy1, gx2, gy2 = gt_box
            gt_box_area = (gx2 - gx1) * (gy2 - gy1)

            inter_x1 = max(x1, gx1)
            inter_y1 = max(y1, gy1)
            inter_x2 = min(x2, gx2)
            inter_y2 = min(y2, gy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = pred_box_area + gt_box_area - inter_area

            iou = inter_area / union_area

            if iou > best_iou:
                best_iou = iou
                best_class = cls

    return best_class

def get_taxonomy_hierarchy(taxonomy, class_name):
    """
    Retrieve the taxonomy hierarchy for a given class name.

    :param taxonomy: An instance of TaxonomyManager.
    :param class_name: The name of the class.
    :return: A list of taxon names representing the hierarchy.
    """
    taxon = taxonomy.get_taxon(class_name)
    hierarchy = []
    while taxon:
        hierarchy.append(f"{taxon.rank}: {taxon.name}")
        taxon = taxon.parent
    return hierarchy[::-1]  # Reverse the list to start from the top of the hierarchy

# Function to append DataFrame to a Parquet file
def append_to_parquet_file(df, parquet_file_path):
    if os.path.exists(parquet_file_path):
        # Read existing data and append new data
        existing_df = pd.read_parquet(parquet_file_path)
        updated_df = pd.concat([existing_df, df])
    else:
        # If file does not exist, use the current DataFrame
        updated_df = df
    updated_df.to_parquet(parquet_file_path, index=False)


# Function to process a single result from YOLO
def process_result(result, basedir, class_index_to_name, use_masks):
    data = []
    orig_img = result.orig_img
    img_shape = orig_img.shape

    if result.boxes is not None:
        for idx, box in enumerate(result.boxes):
            cropped_img = apply_mask_to_detected_object(orig_img, box, result.masks[idx] if use_masks and result.masks is not None else None, use_masks)
            if cropped_img is None:
                continue

            original_filename_stem = Path(result.path).stem
            new_filename = f"{original_filename_stem}_{idx}.jpg"
            path = os.path.join(basedir, new_filename)
            cv2.imwrite(path, cropped_img)

            conf = box.conf.item()
            pred = box.cls.item()
            species_name = class_index_to_name[int(pred)]

            if use_masks and result.masks is not None:
                predicted_mask_xyn = result.masks[idx].xyn[0]
                best_class = find_best_ground_truth_match(result, predicted_mask_xyn=predicted_mask_xyn, img_shape=img_shape)
            else:
                bbox = box.xyxy[0].cpu().numpy()
                best_class = find_best_ground_truth_match(result, bbox=bbox, img_shape=img_shape)

            true_species_name = class_index_to_name[best_class] if best_class is not None else "Unknown"

            entry = {
                'masked_image': path,
                'confidence': conf,
                'predicted_species': species_name,
                'species': true_species_name
            }

            data.append(entry)
    else:
        print("No detections in this image.")
    return data

def maxdepth(tree):
    maxdepth = 0
    for pre, _, node in RenderTree(tree):
        depth = node.depth
        if depth > maxdepth:
            maxdepth = depth
    return maxdepth

# Define get_hierarchical_labels function
def get_hierarchical_labels(species_index, species_names, genus_names, class_names, binary_names, root):
    if species_index == -1:
        return -1, -1, -1  # Handle cases where species_index is invalid

    species_name = species_names[species_index]
    node = next((n for n in root.descendants if n.name == species_name), None)

    if node is None:
        return -1, -1, -1  # Species not found in the tree

    genus_index, class_index, binary_index = -1, -1, -1
    current_node = node
    while current_node.parent is not None:
        current_node = current_node.parent
        if current_node.rank == 'genus':
            genus_index = genus_names.index(current_node.name) if current_node.name in genus_names else -1
        elif current_node.rank == 'class':
            class_index = class_names.index(current_node.name) if current_node.name in class_names else -1
        elif current_node.rank == 'binary':
            binary_index = binary_names.index(current_node.name) if current_node.name in binary_names else -1

    return species_index, genus_index, class_index, binary_index