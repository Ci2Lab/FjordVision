import argparse
import pandas as pd
from anytree.importer import JsonImporter
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict
import sys
import os

# Adding the project directory to the system path
sys.path.append('.')

# Update the import statement to the correct path
from models.ablations.increased_features_complexity.hierarchical_cnn import HierarchicalCNN
from utils.custom_dataset import CustomDataset
from utils.hierarchical_loss import HierarchicalCrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Hierarchical Classification Training')
parser.add_argument('--alpha', type=float, required=True, help='Alpha value for the model')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the model')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
args = parser.parse_args()

# Load taxonomy and class information
importer = JsonImporter()
with open('datasets/ontology.json', 'r') as f:
    root = importer.read(f)

# Paths for class files
classes_file = 'datasets/EMVSD/EMVSD/classes.txt'
object_names = [line.strip() for line in open(classes_file, 'r')]

# Classify nodes by rank
subcategory_names, category_names, binary_names = [], [], []
for node in root.descendants:
    if node.rank == 'genus':
        subcategory_names.append(node.name)
    elif node.rank == 'class':
        category_names.append(node.name)
    elif node.rank == 'binary':
        binary_names.append(node.name)

# Load dataset
df = pd.read_parquet('datasets/yolov8-segmented-objects-dataset.parquet')
train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.5, random_state=42)

# Model Instantiation
rank_counts = defaultdict(int)
for node in root.descendants:
    rank_counts[node.rank] += 1

num_classes_hierarchy = [rank_counts['binary'], rank_counts['class'], rank_counts['genus'], rank_counts['species']]
num_additional_features = 2  # Ensure this is defined as required
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Properly passing both required arguments to the HierarchicalCNN constructor
model = HierarchicalCNN(num_classes_hierarchy, num_additional_features, dropout_rate=args.dropout_rate).to(device)

# Data loaders
train_dataset = CustomDataset(train_df, object_names, subcategory_names, category_names, binary_names, root)
val_dataset = CustomDataset(val_df, object_names, subcategory_names, category_names, binary_names, root)
test_dataset = CustomDataset(test_df, object_names, subcategory_names, category_names, binary_names, root)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Alpha value from argument
initial_alpha = args.alpha
alpha_learnable = False

# Training Preparation
num_epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0
criterion = HierarchicalCrossEntropyLoss(num_levels=len(num_classes_hierarchy), device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Set up directories for logs and weights
log_directory = 'logs/ablations/increased_features_complexity'
weights_directory = 'datasets/hierarchical-model-weights/ablations/increased_features_complexity/weights'
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

        optimizer.zero_grad()
        for i, (images, conf, pred_species, species_index, genus_index, class_index, binary_index) in enumerate(train_loader):
            images, conf, pred_species = images.to(device), conf.to(device), pred_species.to(device)
            species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)

            outputs = model(images, conf, pred_species)
            targets = [binary_index, class_index, genus_index, species_index]
            loss = criterion(outputs, targets)
            loss.backward()
            
            if (i + 1) % args.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, conf, pred_species, species_index, genus_index, class_index, binary_index in val_loader:
                images, conf, pred_species = images.to(device), conf.to(device), pred_species.to(device)
                species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)

                outputs = model(images, conf, pred_species)
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
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)
        torch.save(model.state_dict(), last_model_filename)
