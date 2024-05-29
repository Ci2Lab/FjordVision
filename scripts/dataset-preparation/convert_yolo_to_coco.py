import os
import json
from PIL import Image
from tqdm import tqdm
import supervision as sv
import numpy as np

def read_classes(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return [{'id': i, 'name': class_name} for i, class_name in enumerate(classes)]

def convert_yolo_to_coco_segmentation(image_dir, label_dir, output_path, categories):
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': categories
    }

    annotation_id = 1
    image_id = 1  # Initialize image_id counter
    for image_file in tqdm(sorted(os.listdir(image_dir)), desc="Processing images"):
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        
        coco_data['images'].append({
            'id': image_id,
            'file_name': image_file,
            'width': image.width,
            'height': image.height
        })
        
        label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        if not os.path.exists(label_file):
            continue
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.split()
                class_id = int(parts[0])
                polygon = [float(p) for p in parts[1:]]
                if len(polygon) < 6:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                polygon = [(polygon[i] * image.width if i % 2 == 0 else polygon[i] * image.height) for i in range(len(polygon))]
                
                # Ensure the polygon is in the correct format for bounding box conversion
                polygon_coords = np.array(polygon).reshape(-1, 2)
                
                # Calculate bounding box from polygon
                bounding_box = sv.polygon_to_xyxy(polygon_coords)
                x_coords = polygon_coords[:, 0]
                y_coords = polygon_coords[:, 1]
                bbox_x = min(x_coords)
                bbox_y = min(y_coords)
                bbox_width = max(x_coords) - bbox_x
                bbox_height = max(y_coords) - bbox_y
                
                if bbox_width > 0 and bbox_height > 0:
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': class_id,
                        'bbox': [bbox_x, bbox_y, bbox_width, bbox_height],
                        'area': bbox_width * bbox_height,
                        'iscrowd': 0,
                        'segmentation': [polygon]
                    })
                    annotation_id += 1
        
        image_id += 1  # Increment image_id for each image
    
    with open(output_path, 'w') as outfile:
        json.dump(coco_data, outfile, indent=4)

if __name__ == "__main__":
    # Read categories from classes.txt
    classes_path = 'datasets/EMVSD/EMVSD/classes.txt'
    categories = read_classes(classes_path)

    # Paths to your dataset directories
    dataset_base_path = 'datasets/EMVSD/EMVSD'
    image_dirs = {
        'train': os.path.join(dataset_base_path, 'images/train'),
        'val': os.path.join(dataset_base_path, 'images/val'),
        'test': os.path.join(dataset_base_path, 'images/test')
    }
    label_dirs = {
        'train': os.path.join(dataset_base_path, 'labels/train'),
        'val': os.path.join(dataset_base_path, 'labels/val'),
        'test': os.path.join(dataset_base_path, 'labels/test')
    }

    # Convert each split
    for split in ['train', 'val', 'test']:
        output_path = os.path.join('datasets/EMVSD/EMVSD', f'{split}_annotations.json')
        convert_yolo_to_coco_segmentation(image_dirs[split], label_dirs[split], output_path, categories)
