import torch
import torch.nn as nn
from .branch_cnn import BranchCNN

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features, output_size=(5, 5)):
        super(HierarchicalCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 30, kernel_size=5, padding=2),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(30, 60, kernel_size=5, padding=2),
                nn.BatchNorm2d(60),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(60, 100, kernel_size=3, padding=1),
                nn.BatchNorm2d(100),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(100, 150, kernel_size=3, padding=1),
                nn.BatchNorm2d(150),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])

        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)

        # Initialize branches with a standardized feature size
        self.branches = nn.ModuleList([
            BranchCNN(output_size[0] * output_size[1] * 150, num_classes, num_additional_features)  # Adjust the size accordingly
            for num_classes in num_classes_hierarchy
        ])

        self.activations = []

    def register_hooks(self):
        def hook_fn(module, input, output):
            self.activations.append(output)
            
        # Register the hook on the last layer of each nn.Sequential block
        for block in self.conv_layers:
            # Assuming the last layer of each block is what we're interested in
            last_layer = block[-1]  # Get the last layer in each sequential block
            last_layer.register_forward_hook(hook_fn)

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

        x = self._forward_features(x)

        # Apply adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten the features after adaptive pooling
        x = x.view(x.size(0), -1)

        for branch in self.branches:
            branch_output = branch(x, additional_features)
            outputs.append(branch_output)

        return outputs
    
    def get_activations(self, x, layer_indices):
        """
        Get activations from specified layers.
        
        Args:
        x (torch.Tensor): The input tensor.
        layer_indices (list of int): Indices of the layers from which to get activations.
        
        Returns:
        List of torch.Tensor: Activations from the specified layers.
        """
        activations = []
        
        # Helper function to be used as a hook
        def hook_fn(module, input, output):
            activations.append(output)

        # Register hook for each specified layer
        hooks = []
        for idx in layer_indices:
            layer = self.conv_layers[idx][-1]  # Assuming we're interested in the last layer of each block
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Forward pass
        self._forward_features(x)  # This will trigger the hooks and capture activations

        # Remove hooks after use
        for hook in hooks:
            hook.remove()

        return activations