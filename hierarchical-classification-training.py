# %% [markdown]
# # Libraries

# %%
import pandas as pd
from anytree import Node
from anytree.importer import JsonImporter
from preprocessing import *
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import csv

# %% [markdown]
# # Populate Taxonomy

# %%
importer = JsonImporter()
root = importer.read(open('ontology.json', 'r'))

classes_file = '/mnt/RAID/datasets/label-studio/fjord/classes.txt'

species_names = []
with open(classes_file, 'r') as file:
    species_names = [line.strip() for line in file]

genus_names = [node.name for node in root.descendants if node.rank == 'genus']
class_names = [node.name for node in root.descendants if node.rank == 'class']
binary_names = [node.name for node in root.descendants if node.rank == 'binary']

def get_hierarchical_labels(species_index, root):
    if species_index == -1:
        return -1, -1, -1  # Handle cases where species_index is invalid

    species_name = species_names[species_index]
    node = next((n for n in root.descendants if n.name == species_name), None)

    if node is None:
        return -1, -1, -1  # Species not found in the tree

    # Traverse up the tree to find genus, class, and binary ranks
    genus_index, class_index, binary_index = -1, -1, -1
    current_node = node

    while current_node.parent is not None:
        current_node = current_node.parent
        if current_node.rank == 'genus':
            genus_index = genus_names.index(current_node.name)
        elif current_node.rank == 'class':
            class_index = class_names.index(current_node.name)
        elif current_node.rank == 'binary':
            binary_index = binary_names.index(current_node.name)

    return genus_index, class_index, binary_index

# %% [markdown]
# # Read Dataset

# %%
df = pd.read_parquet('/mnt/RAID/projects/FjordVision/segmented-objects-dataset.parquet')
# Assuming df is your DataFrame with all data
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# %% [markdown]
# # Pytorch Model

# %%
class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, num_additional_features):
        super(BranchCNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(num_in_features + num_additional_features, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, additional_features):
        x = x.view(x.size(0), -1)
        combined_input = torch.cat((x, additional_features), dim=1)
        return self.fc_layers(combined_input)


class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features):
        super(HierarchicalCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])

        # Branches for each convolutional layer output
        self.branches = nn.ModuleList([
            BranchCNN(128 * 64 * 64, num_classes_hierarchy[0], num_additional_features),  # Update sizes accordingly
            BranchCNN(256 * 32 * 32, num_classes_hierarchy[1], num_additional_features),
            BranchCNN(512 * 16 * 16, num_classes_hierarchy[2], num_additional_features),
            BranchCNN(512 * 8 * 8, num_classes_hierarchy[3], num_additional_features)
        ])

    def forward(self, x, conf, iou, pred_species):
        outputs = []
        additional_features = torch.cat((conf.view(-1, 1), iou.view(-1, 1), pred_species.view(-1, 1)), dim=1)

        for conv_layer, branch in zip(self.conv_layers, self.branches):
            x = conv_layer(x)
            branch_output = branch(x, additional_features)
            outputs.append(branch_output)

        return outputs


# Create a defaultdict to store the counts for each rank
rank_counts = defaultdict(int)

# Iterate over the nodes of the tree
for node in root.descendants:
    rank = node.rank
    rank_counts[rank] += 1

# Example instantiation of the model
num_classes_hierarchy = list(rank_counts.values())  # Example: [num_species, num_genus, num_class, num_binary]
num_additional_features = 3  # Assuming 3 additional features: conf, iou, pred_species

model = HierarchicalCNN(num_classes_hierarchy, num_additional_features)

# %%
image = Image.open(df['masked_image'].iloc[0])
image.resize((128, 128))

# %% [markdown]
# # Dataloader

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe, species_names, genus_names, class_names, binary_names):
        self.dataframe = dataframe
        self.species_names = species_names
        self.genus_names = genus_names
        self.class_names = class_names
        self.binary_names = binary_names

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['masked_image']
        # read image from path
        image = Image.open(image_path)
        image_resized = image.resize((128, 128))


        # Convert images to tensor
        image_tensor = torch.tensor(np.array(image_resized), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize if necessary

        # Convert additional features to tensor
        conf_tensor = torch.tensor(row['confidence'], dtype=torch.float32)
        iou_tensor = torch.tensor(row['iou_with_best_gt'], dtype=torch.float32)
  
        # Convert predicted species name to index
        pred_species_index = self.species_names.index(row['predicted_species']) if row['predicted_species'] in self.species_names else -1

        # Convert species names to indices and get hierarchical labels
        species_index = self.species_names.index(row['species']) if row['species'] in self.species_names else -1
        genus_index, class_index, binary_index = get_hierarchical_labels(species_index, root)

        return image_tensor, conf_tensor, iou_tensor, torch.tensor(pred_species_index), torch.tensor(species_index), torch.tensor(genus_index), torch.tensor(class_index), torch.tensor(binary_index)


# Instantiate CustomDataset with the species names list
train_dataset = CustomDataset(train_df, species_names, genus_names, class_names, binary_names)
val_dataset = CustomDataset(val_df, species_names, genus_names, class_names, binary_names)
test_dataset = CustomDataset(test_df, species_names, genus_names, class_names, binary_names)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# %% [markdown]
# # Hierarchical Cross Entropy loss

# %%
class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        total_loss = 0
        for output, target, weight in zip(outputs, targets, self.weights):
            valid_indices = (target >= 0) & (target < output.size(1))  # Ensure target is within range
            if valid_indices.any():
                loss = nn.CrossEntropyLoss()(output[valid_indices], target[valid_indices])
                total_loss += weight * loss
        return total_loss

# Calculate weights based on their normalised relative size of the sum
# of the number of classes at each hierarchical level
weights = [num_classes / sum(num_classes_hierarchy) for num_classes in num_classes_hierarchy]
criterion = HierarchicalCrossEntropyLoss(weights)

# %% [markdown]
# # Training Loop

# %%
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 100
best_val_loss = float('inf')
last_model_path = 'last_model.pth'
best_model_path = 'best_model.pth'

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
        writer.writerow([epoch+1, train_loss, val_loss])

        # Save last model
        torch.save(model.state_dict(), last_model_path)

        # Check if this is the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
