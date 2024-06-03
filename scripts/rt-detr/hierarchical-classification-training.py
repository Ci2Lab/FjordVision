import argparse
import pandas as pd
from anytree.importer import JsonImporter
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict
import sys

# Adding the project directory to the system path
sys.path.append('.')

from models.hierarchical_cnn import HierarchicalCNN
from utils.custom_dataset import CustomDataset
from utils.hierarchical_loss import HierarchicalCrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Hierarchical Classification Training')
parser.add_argument('--alpha', type=float, required=True, help='Alpha value for the model')
args = parser.parse_args()

# Populate Taxonomy
importer = JsonImporter()
with open('datasets/ontology.json', 'r') as f:
    root = importer.read(f)

classes_file = 'datasets/EMVSD/EMVSD/classes.txt'

object_names = []
with open(classes_file, 'r') as file:
    object_names = [line.strip() for line in file]

subcategory_names, category_names, binary_names = [], [], []
for node in root.descendants:
    if node.rank == 'genus':
        subcategory_names.append(node.name)
    elif node.rank == 'class':
        category_names.append(node.name)
    elif node.rank == 'binary':
        binary_names.append(node.name)

# Read Dataset
df = pd.read_parquet('datasets/rtdetr-segmented-objects-dataset.parquet')

train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.5, random_state=42)

# Model Instantiation
rank_counts = defaultdict(int)
for node in root.descendants:
    rank_counts[node.rank] += 1

num_classes_hierarchy = [rank_counts['binary'], rank_counts['class'], rank_counts['genus'], rank_counts['species']]
num_additional_features = 3
num_levels = len(num_classes_hierarchy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalCNN(num_classes_hierarchy, num_additional_features).to(device)

# DataLoader
train_dataset = CustomDataset(train_df, object_names, subcategory_names, category_names, binary_names, root)
val_dataset = CustomDataset(val_df, object_names, subcategory_names, category_names, binary_names, root)
test_dataset = CustomDataset(test_df, object_names, subcategory_names, category_names, binary_names, root)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=8)


# Alpha value from argument
initial_alpha = args.alpha
alpha_learnable = False

# Training Preparation
num_epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0

# Define the criterion
criterion = HierarchicalCrossEntropyLoss(num_levels=len(num_classes_hierarchy), alpha=initial_alpha, learnable_alpha=alpha_learnable, device=device)

# Combine parameters from both model and criterion if criterion has learnable parameters
all_parameters = list(model.parameters()) + list(criterion.parameters()) if alpha_learnable else model.parameters()

# Initialize the AdamW optimizer with both model and potentially criterion's parameters
optimizer = torch.optim.AdamW(all_parameters, lr=0.001, weight_decay=0.01)

# Adjust the scheduler's milestones considering the usual early stopping point
# Inside your training script, after initializing the optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Dynamic file names based on alpha value
log_filename = f'logs/model_alpha_{args.alpha:.2f}.csv'
best_model_filename = f'datasets/hierarchical-model-weights/weights/best_model_alpha_rtdetr_{args.alpha:.2f}.pth'
last_model_filename = f'datasets/hierarchical-model-weights/weights/last_model_alpha_rtdetr_{args.alpha:.2f}.pth'

# Training and Validation Loop with Logging
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = ['Epoch', 'Training Loss', 'Validation Loss', 'Alpha'] + [f'Lambda Weight Lvl {i+1}' for i in range(num_levels)]
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

        lambda_weights_list = criterion.lambda_weights.cpu().detach().numpy().tolist()
        log_row = [epoch+1, train_loss, val_loss, criterion.alpha.item()] + lambda_weights_list
        writer.writerow(log_row)
        print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Alpha: {criterion.alpha.item():.4f}, Lambda Weights: {lambda_weights_list}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_filename)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # After computing validation loss at the end of an epoch
        scheduler.step(val_loss)

        torch.save(model.state_dict(), last_model_filename)