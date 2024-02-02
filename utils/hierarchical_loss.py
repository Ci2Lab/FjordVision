import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCrossEntropyLoss(nn.Module):
    """
    A custom loss function that computes the cross-entropy loss for multiple hierarchical levels,
    applying different weights to each level's loss. It filters out invalid target indices before
    loss computation.
    """
    def __init__(self, weights):
        """
        Initializes the HierarchicalCrossEntropyLoss instance.

        :param weights: A list of weights for each hierarchical level's loss contribution
                        to the total loss.
        """
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        """
        Computes the weighted hierarchical cross-entropy loss.

        :param outputs: A list of tensor predictions from the model, one for each hierarchical level.
        :param targets: A list of tensor labels, one for each hierarchical level.
        :return: The total loss as a scalar tensor.
        """
        total_loss = 0.0
        for output, target, weight in zip(outputs, targets, self.weights):
            # Ensure target labels are within the valid range of the current output predictions
            valid_indices = (target >= 0) & (target < output.size(1))
            if valid_indices.any():
                # Compute the cross-entropy loss for valid indices only
                loss = F.cross_entropy(output[valid_indices], target[valid_indices])
                # Weight the loss and add to the total loss
                total_loss += weight * loss
        return total_loss
