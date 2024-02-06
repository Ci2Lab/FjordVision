import torch
import torch.nn as nn
from .branch_cnn import BranchCNN

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features):
        super(HierarchicalCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])

        # Calculate feature sizes for each branch dynamically
        # Example usage in __init__:
        self.feature_sizes = [self._get_conv_output((3, 128, 128), i) for i, _ in enumerate(self.conv_layers)]

        self.branches = nn.ModuleList([
            BranchCNN(feature_size, num_classes, num_additional_features)
            for feature_size, num_classes in zip(self.feature_sizes, num_classes_hierarchy)
        ])


    def _get_conv_output(self, shape, conv_layer_end):
        """
        Calculates the input feature size for a linear layer after a specific convolutional block.
        """
        input = torch.autograd.Variable(torch.rand(1, *shape))
        for i, layer in enumerate(self.conv_layers):
            input = layer(input)
            if i == conv_layer_end:  # Stop after the specified block
                break
        n_size = input.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        """
        Forward pass through the convolutional layers to dynamically calculate output feature size.
        """
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x

    def forward(self, x, conf, iou, pred_species):
        outputs = []
        additional_features = torch.cat((conf.view(-1, 1), iou.view(-1, 1), pred_species.view(-1, 1)), dim=1)

        for conv_layer, branch in zip(self.conv_layers, self.branches):
            x = conv_layer(x)
            branch_output = branch(x, additional_features)
            outputs.append(branch_output)

        return outputs

