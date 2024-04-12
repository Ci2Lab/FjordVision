import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, num_additional_features):
        super(BranchCNN, self).__init__()
        
        # Calculating the combined features size
        combined_features_size = num_in_features + num_additional_features * 2
        
        # Fully connected layers for processing combined features
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_features_size, 1024),
            nn.BatchNorm1d(1024),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        # Additional layers for processing the extra features
        self.additional_feature_layers = nn.Sequential(
            nn.Linear(num_additional_features, num_additional_features * 2),
            Mish(),
            nn.Linear(num_additional_features * 2, num_additional_features * 2),
            Mish(),
        )

    def forward(self, x, additional_features):
        # Flatten the input features
        x = x.view(x.size(0), -1)

        # Process additional features
        processed_additional_features = self.additional_feature_layers(additional_features)

        # Combine processed additional features with the main features
        combined_input = torch.cat((x, processed_additional_features), dim=1)

        # Pass the combined input through the fully connected layers
        return self.fc_layers(combined_input)
