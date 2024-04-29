import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import functional as F
from anytree.importer import JsonImporter
from collections import defaultdict
import random
import sys
import os

sys.path.append('.')
from preprocessing.preprocessing import find_best_ground_truth_match
from models.hierarchical_cnn import HierarchicalCNN

# Load Taxonomy
def load_taxonomy(ontology_path, classes_path):
    importer = JsonImporter()
    with open(ontology_path, 'r') as f:
        root = importer.read(f)
    
    # Load the ordering of species from classes.txt
    with open(classes_path, 'r') as file:
        ordered_species = [line.strip() for line in file]

    labels = {node.rank: [] for node in root.descendants if node.rank}
    
    # Update species to follow the order defined in classes.txt
    species_set = set(ordered_species)  # Convert list to set for fast lookup
    labels['species'] = [species for species in ordered_species if species in species_set]

    # Append remaining species that might not be in classes.txt
    additional_species = [node.name for node in root.descendants if node.rank == 'species' and node.name not in species_set]
    labels['species'].extend(additional_species)
    
    # Load other ranks
    for node in root.descendants:
        if node.rank != 'species':
            labels[node.rank].append(node.name)

    return labels

def calculate_num_classes(ontology_path):
    with open(ontology_path, 'r') as f:
        root = JsonImporter().read(f)
    rank_counts = defaultdict(int)
    for node in root.descendants:
        rank_counts[node.rank] += 1
    return rank_counts  # Return a dictionary directly

# Assign Colors
def assign_colors(labels):
    unique_labels = set(labels)
    colors = {}
    for label in unique_labels:
        colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors

# Load Models
def load_model(model_path, model_type='hierarchical', device='cuda', num_classes=None):
    print("Resolved model path:", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model path {model_path} does not exist.")
    if model_type == 'yolo':
        print("Loading YOLO model...")
        model = YOLO(model_path)  # Load the model in inference mode
    else:
        print("Loading Hierarchical CNN model...")
        num_additional_features = 3
        model = HierarchicalCNN(num_classes, num_additional_features)
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

# Preprocess Image
def preprocess_image(image):
    resized_image = cv2.resize(image, (640, 640))
    resized_image = F.to_tensor(resized_image).unsqueeze_(0).to('cuda')
    return resized_image.float()

def main(video_path, yolo_model_path, hierarchical_model_path, ontology_path, classes_file, display_level='species'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels_hierarchy = load_taxonomy(ontology_path, classes_file)
    label_colors = assign_colors([label for sublist in labels_hierarchy.values() for label in sublist])

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('demo/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    rank_counts = calculate_num_classes(ontology_path)
    num_classes_hierarchy = [rank_counts.get('binary', 0), rank_counts.get('class', 0), rank_counts.get('genus', 0), rank_counts.get('species', 0)]
    
    yolo_model = load_model(yolo_model_path, 'yolo', device)
    hierarchical_model = load_model(hierarchical_model_path, 'hierarchical', device, num_classes_hierarchy)
    hierarchical_model.eval()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with the YOLO model
        results = yolo_model(frame)

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for idx, mask in enumerate(result.masks):
                    predictions = {level: [] for level in ['binary', 'class', 'genus', 'species']}
                    mask_data = mask.data.cpu().numpy()
                    mask_resized = cv2.resize((mask_data[0] > 0.5).astype(np.uint8), (frame.shape[1], frame.shape[0]))
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # Draw each contour with a transparent color
                    overlay = frame.copy()
                    alpha = 0.5  # Transparency factor

                    # Fill the contour with color based on class
                    for contour in contours:
                        cls_label = labels_hierarchy[display_level][idx]
                        color = label_colors[cls_label]
                        cv2.fillPoly(overlay, [contour], color)

                    # Blend the overlay with the original frame
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    mask_img = frame * (mask_resized[:, :, None].astype(frame.dtype) / 255)
                    mask_tensor = preprocess_image(mask_img)
                    conf = torch.tensor([result.boxes.conf[idx].item()], device=device).float()
                    cls_label = torch.tensor([result.boxes.cls[idx].item()], device=device).float()
                    default_iou = torch.tensor([0.5], device=device).float()
                    hierarchical_output = hierarchical_model(mask_tensor, conf, default_iou, cls_label)
                    
                    for i, output in enumerate(hierarchical_output):
                        _, predicted = torch.max(output, 1)
                        level = ['binary', 'class', 'genus', 'species'][i]
                        predictions[level].extend(predicted.cpu().numpy())
                    
                    label = predictions[display_level][0]

                    if isinstance(label, list):  # Ensure label is a single integer
                        label = label[0] if label else -1  # Use -1 if list is empty

                    # Convert ontology label to YOLO label
                    label_name = labels_hierarchy[display_level][label]

                    # Find centroid of the mask
                    M = cv2.moments(mask_resized)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, str(label_name), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)  # Save the processed frame to the output video

    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = "demo/demo.mp4"
    yolo_model_path = "datasets/pre-trained-models/fjord/Yolov8n-seg.pt"
    hierarchical_model_path = "datasets/hierarchical-model-weights/weights/best_model_alpha_0.80.pth"
    ontology_path = "datasets/ontology.json"
    classes_file = "datasets/The Fjord Dataset/fjord/classes.txt"
    main(video_path, yolo_model_path, hierarchical_model_path, ontology_path, classes_file)
