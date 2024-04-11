import torch
import torch.nn as nn

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, num_additional_features):
        super(BranchCNN, self).__init__()
        
        # Adjust the input size to account for the combined features
        combined_features_size = num_in_features + num_additional_features * 2  # Feature expansion
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_features_size, 1024),
            nn.BatchNorm1d(1024),  # Adding Batch Normalization
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # Adding Batch Normalization
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Adding Batch Normalization
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        # Enhanced additional feature layers
        self.additional_feature_layers = nn.Sequential(
            nn.Linear(num_additional_features, num_additional_features * 2),  # Expansion
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(num_additional_features * 2, num_additional_features * 2),  # Maintaining the expanded size
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x, additional_features):
        x = x.view(x.size(0), -1)
        
        # Process additional features separately and expand
        processed_additional_features = self.additional_feature_layers(additional_features)
        
        # Combine the main and additional features
        combined_input = torch.cat((x, processed_additional_features), dim=1)
        return self.fc_layers(combined_input)
