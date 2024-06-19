import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, num_levels, alpha=0.5, learnable_alpha=True, device='cuda'):
        """
        Initializes the HierarchicalCrossEntropyLoss instance with an optional learnable alpha parameter.

        :param num_levels: The number of hierarchical levels.
        :param alpha: The initial value for the alpha parameter.
        :param learnable_alpha: Specifies whether alpha should be a learnable parameter.
        :param device: The device (cpu or cuda) where the tensors will be allocated.
        """
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.device = device
        self.num_levels = num_levels
        
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float, device=device))
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float, device=device)
        
        self.update_lambda_weights()  # Initialize lambda weights based on initial alpha
        
    def update_lambda_weights(self):
        """Update lambda weights based on the current value of alpha."""
        self.lambda_weights = torch.exp(-self.alpha * (self.num_levels - 1 - torch.arange(self.num_levels, device=self.device)))
        
    def forward(self, outputs, targets):
        """
        Computes the weighted hierarchical cross-entropy loss with dynamic hierarchical emphasis.

        :param outputs: A list of tensor predictions from the model, one for each hierarchical level.
        :param targets: A list of tensor labels, one for each hierarchical level.
        :return: The total loss as a scalar tensor.
        """
        self.update_lambda_weights()  # Ensure lambda weights are up-to-date
        
        total_loss = 0.0
        for h, (output, target) in enumerate(zip(outputs, targets)):
            valid_indices = (target >= 0) & (target < output.size(1))
            if valid_indices.any():
                loss = F.cross_entropy(output[valid_indices], target[valid_indices])
                total_loss += self.lambda_weights[h] * loss
        return total_loss
