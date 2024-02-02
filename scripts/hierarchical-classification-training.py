# hierarchical_classification_training.py
import pandas as pd

# Libraries
import pandas as pd
from anytree.importer import JsonImporter
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict

# Custom imports based on the new structure
from models.hierarchical_cnn import HierarchicalCNN
from utils.custom_dataset import CustomDataset  # Adjust the import path as necessary
from utils.hierarchical_loss import HierarchicalCrossEntropyLoss  # Adjust the import path as necessary
from preprocessing.preprocessing import get_hierarchical_labels

# Populate Taxonomy
importer = JsonImporter()
root = importer.read(open('data/ontology.json', 'r'))

classes_file = '/mnt/RAID/datasets/label-studio/fjord/classes.txt'

species_names = []
with open(classes_file, 'r') as file:
    species_names = [line.strip() for line in file]

genus_names, class_names, binary_names = [], [], []
for node in root.descendants:
    if node.rank == 'genus':
        genus_names.append(node.name)
    elif node.rank == 'class':
        class_names.append(node.name)
    elif node.rank == 'binary':
        binary_names.append(node.name)

# Read Dataset
df = pd.read_parquet('data/segmented-objects-dataset.parquet')

train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# Model Instantiation
# Create a defaultdict to store the counts for each rank
rank_counts = defaultdict(int)

# Iterate over the nodes of the tree
for node in root.descendants:
    rank_counts[node.rank] += 1

num_classes_hierarchy = [rank_counts['binary'], rank_counts['class'], rank_counts['genus'], rank_counts['species']]
num_additional_features = 3

model = HierarchicalCNN(num_classes_hierarchy, num_additional_features).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# DataLoader
train_dataset = CustomDataset(train_df, species_names, genus_names, class_names, binary_names, root)
val_dataset = CustomDataset(val_df, species_names, genus_names, class_names, binary_names, root)
test_dataset = CustomDataset(test_df, species_names, genus_names, class_names, binary_names, root)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Hierarchical Cross Entropy Loss
weights = [0, 0, 0, 1]  # Example weights, adjust as necessary
criterion = HierarchicalCrossEntropyLoss(weights)

# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 100
best_val_loss = float('inf')
last_model_path = 'models/weights/last_model.pth'
best_model_path = 'models/weights/best_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open a file to log the losses
with open('training_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])

    for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for images, conf, iou, pred_species, species_index, genus_index, class_index, binary_index in train_loader:
                images, conf, iou, pred_species = images.to(device), conf.to(device), iou.to(device), pred_species.to(device)
                species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)

                optimizer.zero_grad()
                outputs = model(images, conf, iou, pred_species)
                targets = [species_index, genus_index, class_index, binary_index]  # Update this as per your model's requirements
                loss = criterion(outputs, targets)  # Make sure this matches your model and data
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, conf, iou, pred_species, species_index, genus_index, class_index, binary_index in val_loader:
                    images, conf, iou, pred_species = images.to(device), conf.to(device), iou.to(device), pred_species.to(device)
                    species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)

                    outputs = model(images, conf, iou, pred_species)
                    loss = criterion(outputs, [species_index, genus_index, class_index, binary_index])
                    val_loss += loss.item()

            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            writer.writerow([epoch+1, train_loss, val_loss])

            # Save last model
            torch.save(model.state_dict(), last_model_path)

            # Check if this is the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)