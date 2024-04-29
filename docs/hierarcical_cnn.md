# HierarchicalCNN and Attention Modules Documentation

This document provides detailed code documentation for the `HierarchicalCNN` model and its related attention mechanisms used in the FjordVision project. These components are crucial for advanced feature extraction and hierarchical classification.

## Modules Documentation

### Mish Activation Function

The `Mish` class implements a custom activation function that enhances performance in deep neural networks by maintaining a smooth gradient flow.

```python
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))
```

### Channel Attention Module

The ChannelAttention module aims to enhance the model's feature representation by applying channel-wise attention, which emphasizes important features.

```python
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            Mish(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return nn.Sigmoid()(out).view(x.size(0), x.size(1), 1, 1) * x

```

### Spatial Attention Module

The SpatialAttention module is designed to enhance the spatial focus of the network on relevant features by applying a sigmoid-activated convolution over combined maximum and average pooled channels

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(combined))
        return x * attention.expand_as(x)
```

### HierarcicalCNN Model

```python
import torch
from models.hierarchical_cnn import HierarchicalCNN
from utils.custom_dataset import CustomDataset
from torch.utils.data import DataLoader

# Define the hierarchical structure for the number of classes at each level
num_classes_hierarchy = [2, 5, 10, 20]  # Binary, Class, Genus, Species levels
num_additional_features = 3  # Number of additional features like confidence or IOU

# Initialize the HierarchicalCNN model
model = HierarchicalCNN(num_classes_hierarchy, num_additional_features)

# Prepare a sample dataset
# Note: In a real scenario, replace the random tensors below with actual feature data and metadata.
data = {'features': torch.randn(32, 3, 224, 224),  # Example feature data (batch size, channels, H, W)
        'conf': torch.randn(32, 1),  # Example confidence scores
        'iou': torch.randn(32, 1),  # Example intersection-over-union scores
        'pred_species': torch.randn(32, 20)}  # Predicted probabilities for 20 species

# Properly initialize the dataset with meaningful lists instead of empty ones.
# These lists should contain the names or labels corresponding to categories at each hierarchical level.
object_names = ['object1', 'object2']  # Example object names
subcategory_names = ['subcategory1', 'subcategory2']  # Example subcategory names
category_names = ['category1', 'category2']  # Example category names
binary_names = ['binary1', 'binary2']  # Example binary classification names
tree_root = None  # Replace 'None' with the actual root node of your taxonomy structure if available.

# Instantiate the custom dataset
dataset = CustomDataset(data, object_names, subcategory_names, category_names, binary_names, tree_root)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Fetch a batch of data
for batch in dataloader:
    images, conf, iou, pred_species = batch['features'], batch['conf'], batch['iou'], batch['pred_species']

    # Compute output
    output = model(images, conf, iou, pred_species)
    print(output)  # Output from the model after processing the batch

```