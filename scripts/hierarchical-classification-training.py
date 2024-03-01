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
df = pd.read_parquet('/mnt/RAID/projects/FjordVision/data/segmented-objects-dataset.parquet')

train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.5, random_state=42)

# Model Instantiation
rank_counts = defaultdict(int)
for node in root.descendants:
    rank_counts[node.rank] += 1

num_classes_hierarchy = [rank_counts['binary'], rank_counts['class'], rank_counts['genus'], rank_counts['species']]
num_additional_features = 3

# Define num_levels here based on the length of num_classes_hierarchy
num_levels = len(num_classes_hierarchy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalCNN(num_classes_hierarchy, num_additional_features).to(device)

# DataLoader
train_dataset = CustomDataset(train_df, species_names, genus_names, class_names, binary_names, root)
val_dataset = CustomDataset(val_df, species_names, genus_names, class_names, binary_names, root)
test_dataset = CustomDataset(test_df, species_names, genus_names, class_names, binary_names, root)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Specify the initial value of alpha and whether it should be learnable
initial_alpha = 0.5  # Example initial value
alpha_learnable = False  # Set True if alpha should be learnable, False otherwise

# Training Preparation
num_epochs = 100
best_val_loss = float('inf')
patience = 5
patience_counter = 0
# Update the instantiation with new parameters
criterion = HierarchicalCrossEntropyLoss(num_levels=len(num_classes_hierarchy), alpha=initial_alpha, learnable_alpha=alpha_learnable, device=device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)
scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)

# Logging Setup
with open('logs/model_alpha_05.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Adjust headers to include lambda weights; assume max number of levels is known
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

        # Validation
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
                
        # After training and validation for the epoch, log the details including lambda weights
        lambda_weights_list = criterion.lambda_weights.cpu().detach().numpy().tolist()  # Convert lambda_weights tensor to a list
        log_row = [epoch+1, 
                   train_loss, 
                   val_loss, 
                   criterion.alpha.item()] + lambda_weights_list
        
        writer.writerow(log_row)
        print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Alpha: {criterion.alpha.item():.4f}, Lambda Weights: {lambda_weights_list}")

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/weights/best_model_alpha_05.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

        # Save the last model
        torch.save(model.state_dict(), 'models/weights/last_model_alpha_05.pth')
