# Libraries
import pandas as pd
from anytree.importer import JsonImporter
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict

# Custom imports based on the new structure
from models.hierarchical_cnn import HierarchicalCNN
from utils.custom_dataset import CustomDataset
from utils.hierarchical_loss import HierarchicalCrossEntropyLoss
from preprocessing.preprocessing import get_hierarchical_labels
# Assuming optimizer is defined as in your script
from torch.optim.lr_scheduler import MultiStepLR

# Populate Taxonomy
importer = JsonImporter()
with open('data/ontology.json', 'r') as f:
    root = importer.read(f)

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

train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.5, random_state=42)

# Model Instantiation
rank_counts = defaultdict(int)
for node in root.descendants:
    rank_counts[node.rank] += 1

num_classes_hierarchy = [rank_counts['binary'], rank_counts['class'], rank_counts['genus'], rank_counts['species']]
num_additional_features = 3

model = HierarchicalCNN(num_classes_hierarchy, num_additional_features).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# Before entering the training loop, ensure device is defined correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# DataLoader
train_dataset = CustomDataset(train_df, species_names, genus_names, class_names, binary_names, root)
val_dataset = CustomDataset(val_df, species_names, genus_names, class_names, binary_names, root)
test_dataset = CustomDataset(test_df, species_names, genus_names, class_names, binary_names, root)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Assuming there are 4 hierarchical levels: binary, class, genus, species
num_levels = 4
criterion = HierarchicalCrossEntropyLoss(num_levels=num_levels).to(device)

# Optimizer setup
optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)

# Training
num_epochs = 20
best_val_loss = float('inf')
last_model_path = 'models/weights/flat_last_model.pth'
best_model_path = 'models/weights/flat_best_model.pth'

# Define your scheduler
scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

with open('training_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Learnable Weights'])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, conf, iou, pred_species, species_index, genus_index, class_index, binary_index in train_loader:
            images, conf, iou, pred_species = images.to(device), conf.to(device), iou.to(device), pred_species.to(device)
            species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)

            optimizer.zero_grad()
            outputs = model(images, conf, iou, pred_species)
            targets = [species_index, genus_index, class_index, binary_index]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            criterion.weights.data.clamp_(min=0)
            for images, conf, iou, pred_species, species_index, genus_index, class_index, binary_index in val_loader:
                images, conf, iou, pred_species = images.to(device), conf.to(device), iou.to(device), pred_species.to(device)
                species_index, genus_index, class_index, binary_index = species_index.to(device), genus_index.to(device), class_index.to(device), binary_index.to(device)

                outputs = model(images, conf, iou, pred_species)
                loss = criterion(outputs, [species_index, genus_index, class_index, binary_index])
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        learnable_weights = criterion.weights.detach().cpu().numpy()
        print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        writer.writerow([epoch+1, train_loss, val_loss, learnable_weights])

        scheduler.step()

        torch.save(model.state_dict(), last_model_path)  # Save last model

        if val_loss < best_val_loss:  # Check if this is the best model
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)