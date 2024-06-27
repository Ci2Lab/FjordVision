import os
import json
from PIL import Image
from tqdm import tqdm

# Load the hierarchy
with open('datasets/ontology.json', 'r') as f:
    hierarchy = json.load(f)

def get_class_name(species_name, hierarchy):
    for child in hierarchy['children']:
        for sub_child in child['children']:
            for sub_sub_child in sub_child['children']:
                if 'children' in sub_sub_child:
                    for genus in sub_sub_child['children']:
                        if 'children' in genus:
                            for species in genus['children']:
                                if species['name'] == species_name:
                                    return sub_sub_child['name']
    return None

def update_categories(file_path, hierarchy):
    with open(file_path, 'r') as f:
        coco_data = json.load(f)

    for category in coco_data['categories']:
        species_name = category['name']
        class_name = get_class_name(species_name, hierarchy)
        if class_name:
            category['supercategory'] = class_name
        else:
            category['supercategory'] = 'none'

    with open(file_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Updated categories in {file_path} to include 'supercategory'.")

def yolo_to_coco(yolo_annotation_dir, coco_output_file, image_dir):
    categories = []
    category_set = set()
    images = []
    annotations = []

    annotation_id = 0

    for subset in ['train', 'val', 'test']:
        yolo_dir = os.path.join(yolo_annotation_dir, subset)
        image_dir_subset = os.path.join(image_dir, subset)
        for idx, filename in enumerate(tqdm(os.listdir(yolo_dir))):
            if not filename.endswith('.txt'):
                continue
            
            with open(os.path.join(yolo_dir, filename), 'r') as file:
                lines = file.readlines()
            
            image_id = idx + 1
            image_file = filename.replace('.txt', '.jpg')
            image_path = os.path.join(image_dir_subset, image_file)
            
            # Load image to get its dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            image_info = {
                'id': image_id,
                'file_name': image_file,
                'width': width,
                'height': height
            }
            images.append(image_info)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Skipping invalid annotation in {filename}: {line.strip()}")
                    continue

                category_id = int(parts[0]) + 1  # YOLOv8 class id starts from 0, COCO class id starts from 1
                bbox = [float(p) for p in parts[1:5]]  # Assuming the first 4 values are for bbox
                
                # Convert YOLO bbox format (x_center, y_center, width, height) to COCO format (x_min, y_min, width, height)
                x_center, y_center, w, h = bbox
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                bbox_abs = [x_min, y_min, w * width, h * height]
                
                segmentation = [float(p) for p in parts[5:]] if len(parts) > 5 else []

                if segmentation:
                    # Convert segmentation points to absolute coordinates
                    segmentation_abs = [seg * width if i % 2 == 0 else seg * height for i, seg in enumerate(segmentation)]
                    # Calculate the bounding box from segmentation points
                    x_coords = segmentation_abs[0::2]
                    y_coords = segmentation_abs[1::2]
                    x_min_seg = min(x_coords)
                    y_min_seg = min(y_coords)
                    width_seg = max(x_coords) - x_min_seg
                    height_seg = max(y_coords) - y_min_seg
                    bbox_abs = [x_min_seg, y_min_seg, width_seg, height_seg]

                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': bbox_abs,
                    'segmentation': [segmentation_abs] if segmentation else [],
                    'area': bbox_abs[2] * bbox_abs[3],
                    'iscrowd': 0
                }
                annotations.append(annotation)
                annotation_id += 1
                
                if category_id not in category_set:
                    categories.append({
                        'id': category_id,
                        'name': str(category_id),
                        'supercategory': 'none'
                    })
                    category_set.add(category_id)
    
    coco_output = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(coco_output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"Created COCO annotations at {coco_output_file}")

def main():
    annotation_files = {
        'train': 'datasets/EMVSD/EMVSD/train_annotations.json',
        'val': 'datasets/EMVSD/EMVSD/val_annotations.json',
        'test': 'datasets/EMVSD/EMVSD/test_annotations.json'
    }

    # Check if files exist and create them if they don't
    yolo_annotation_dir = 'datasets/EMVSD/EMVSD/labels'
    image_dir = 'datasets/EMVSD/EMVSD/images'
    
    for subset, file_path in annotation_files.items():
        if not os.path.exists(file_path):
            yolo_to_coco(yolo_annotation_dir, file_path, image_dir)
        else:
            print(f"{file_path} already exists.")

    # Update categories with supercategory
    for file_path in annotation_files.values():
        update_categories(file_path, hierarchy)

if __name__ == "__main__":
    main()