import torch
import torch.nn as nn

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
        logits = self.fc_layers(combined_input)
        return logits  # Removed softmax here for reasons mentioned earlier
