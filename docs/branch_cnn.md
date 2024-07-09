# BranchCNN Model Documentation

This document provides a comprehensive explanation of the `BranchCNN` model used in the FjordVision project, detailing the custom neural network classes defined using PyTorch.

## Modules

### Mish Activation Function

The `Mish` class implements a custom activation function used in neural networks, defined as:

$$
\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))
$$

where `softplus` is a smooth approximation to the ReLU function. This activation function is favored for its properties that help maintain a smooth gradient flow in deeper networks.

#### Implementation

- **Function**: Applies the Mish activation function to the input tensor `x`.
- **Method**: `forward(x)`

### BranchCNN Class

The `BranchCNN` class is designed to efficiently combine primary input features with additional contextual or auxiliary features into a single predictive model.

#### Constructor Parameters

- `num_in_features`: Integer representing the number of input features.
- `num_classes`: Integer representing the number of target output classes.
- `dropout_rate`: Float representing the dropout rate to prevent overfitting. Default is 0.5.

#### Architecture

1. **Fully Connected Layers**:
   - Processes the concatenated primary and additional features.
   - Includes layers for normalization and dropout to prevent overfitting.
   - Employs Mish activation function throughout for non-linearity.

#### Forward Pass

- **Inputs**:
  - `x`: Tensor containing the primary features.
  - `additional_features`: Tensor containing the auxiliary features.
- **Process**:
  - Main and auxiliary features are concatenated and passed through dense layers.
- **Output**:
  - Returns the logits for each class based on the combined feature set.

### Example Usage

Below is a simple example to demonstrate how to instantiate and use the `BranchCNN` model:

```python
import torch
from branch_cnn import BranchCNN

# Initialize model
model = BranchCNN(num_in_features=100, num_classes=10, dropout_rate=0.5)

# Prepare input tensors
x = torch.randn(32, 100)  # Batch size of 32, 100 features each
additional_features = torch.randn(32, 20)  # Batch size of 32, 20 additional features each

# Compute output
output = model(x, additional_features)
