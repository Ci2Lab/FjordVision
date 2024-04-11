import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, num_additional_features):
        super(BranchCNN, self).__init__()
        
        combined_features_size = num_in_features + num_additional_features * 2
        
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
        
        self.additional_feature_layers = nn.Sequential(
            nn.Linear(num_additional_features, num_additional_features * 2),
            Mish(),
            nn.Linear(num_additional_features * 2, num_additional_features * 2),
            Mish(),
        )

    def forward(self, x, additional_features):
        x = x.view(x.size(0), -1)
        processed_additional_features = self.additional_feature_layers(additional_features)
        combined_input = torch.cat((x, processed_additional_features), dim=1)
        return self.fc_layers(combined_input)
