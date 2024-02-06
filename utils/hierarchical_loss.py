import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, num_levels, init_weights=None):
        """
        Initializes the HierarchicalCrossEntropyLoss instance with learnable weights.

        :param num_levels: The number of hierarchical levels.
        :param init_weights: Optional initial weights for each hierarchical level. If not provided,
                             weights will be initialized to 1.
        """
        super(HierarchicalCrossEntropyLoss, self).__init__()
        if init_weights is None:
            init_weights = torch.ones(num_levels)
        # Ensure weights are a parameter to be learned
        self.weights = nn.Parameter(init_weights, requires_grad=True)

    def forward(self, outputs, targets):
        """
        Computes the weighted hierarchical cross-entropy loss with learnable weights.

        :param outputs: A list of tensor predictions from the model, one for each hierarchical level.
        :param targets: A list of tensor labels, one for each hierarchical level.
        :return: The total loss as a scalar tensor.
        """
        total_loss = 0.0
        for i, (output, target) in enumerate(zip(outputs, targets)):
            # Ensure target labels are within the valid range
            valid_indices = (target >= 0) & (target < output.size(1))
            if valid_indices.any():
                loss = F.cross_entropy(output[valid_indices], target[valid_indices])
                # Apply the learnable weight to the loss
                total_loss += self.weights[i] * loss
        return total_loss
