import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import csv
import sys
import os
from sklearn.model_selection import train_test_split
from anytree.importer import JsonImporter
from collections import defaultdict

# Ensure the project directory is in the system path
sys.path.append('/mnt/RAID/projects/FjordVision/')

from models.ablations.increased_features_complexity.hierarchical_cnn import HierarchicalCNN
from utils.custom_dataset import CustomDatasetCoco
from utils.hierarchical_loss import HierarchicalCrossEntropyLoss

parser = argparse.ArgumentParser(description='COCO Hierarchical Classification Training')
parser.add_argument('--alpha', type=float, required=True, help='Alpha value for the model')
args = parser.parse_args()

# Populate Taxonomy
importer = JsonImporter()
with open('/mnt/RAID/projects/FjordVision/data/coco.json', 'r') as f:
    root = importer.read(f)

classes_file = '/mnt/RAID/datasets/coco/classes.txt'

object_names = []
with open(classes_file, 'r') as file:
    object_names = [line.strip() for line in file]


# Classify nodes by category, subcategory, etc., specific to COCO
category_names, subcategory_names, binary_names = [], [], []
for node in root.descendants:
    if node.rank == 'subcategory':
        subcategory_names.append(node.name)
    elif node.rank == 'category':
        category_names.append(node.name)
    elif node.rank == 'binary':
        binary_names.append(node.name)

# Load COCO dataset
df = pd.read_parquet('/mnt/RAID/projects/FjordVision/data/coco-segmented-objects-dataset.parquet')
train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.5, random_state=42)

# Model Instantiation
rank_counts = defaultdict(int)
for node in root.descendants:
    rank_counts[node.rank] += 1

num_classes_hierarchy = [rank_counts['binary'], rank_counts['category'], rank_counts['subcategory'], rank_counts['object']]
num_additional_features = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalCNN(num_classes_hierarchy, num_additional_features).to(device)

# Data loaders for COCO
train_dataset = CustomDatasetCoco(train_df, object_names, category_names, subcategory_names, binary_names, root)
val_dataset = CustomDatasetCoco(val_df, object_names, category_names, subcategory_names, binary_names, root)
test_dataset = CustomDatasetCoco(test_df, object_names, category_names, subcategory_names, binary_names, root)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=8)

# Training Setup
num_epochs = 100
best_val_loss = float('inf')
patience_counter = 0
criterion = HierarchicalCrossEntropyLoss(num_levels=len(num_classes_hierarchy), device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Set up directories for logs and weights specific to COCO ablation
log_directory = '/mnt/RAID/projects/FjordVision/models/ablations/increased_features_complexity/logs-coco/'
weights_directory = '/mnt/RAID/projects/FjordVision/models/ablations/increased_features_complexity/weights-coco/'
os.makedirs(log_directory, exist_ok=True)
os.makedirs(weights_directory, exist_ok=True)

log_filename = os.path.join(log_directory, f'model_alpha_{args.alpha:.2f}.csv')
best_model_filename = os.path.join(weights_directory, f'best_model_alpha_{args.alpha:.2f}.pth')
last_model_filename = os.path.join(weights_directory, f'last_model_alpha_{args.alpha:.2f}.pth')

# Training and Validation Loop
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = ['Epoch', 'Training Loss', 'Validation Loss', 'Alpha']
    writer.writerow(headers)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, conf, iou, pred_species, species_index, genus_index, class_index, binary_index in train_loader:
            images, conf, iou, pred_species = images.to(device), conf.to(device), iou.to(device), pred_species.to(device)
            species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, conf, iou, pred_species)
            targets = [binary_index, class_index, genus_index, species_index]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, conf, iou, pred_species, species_index, genus_index, class_index, binary_index in val_loader:
                images, conf, iou, pred_species = images.to(device), conf.to(device), iou.to(device), pred_species.to(device)
                species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)
                
                outputs = model(images, conf, iou, pred_species)
                targets = [binary_index, class_index, genus_index, species_index]
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        log_row = [epoch+1, train_loss, val_loss, args.alpha]
        writer.writerow(log_row)
        print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Alpha: {args.alpha:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_filename)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)
        torch.save(model.state_dict(), last_model_filename)
