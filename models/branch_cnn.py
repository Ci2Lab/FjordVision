import torch
import torch.nn as nn

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, num_additional_features):
        super(BranchCNN, self).__init__()
        
        # Adjust the input size to account for the combined features from the new architecture
        # Assuming output_size is (5, 5) as defined in HierarchicalCNN, then num_in_features would be 512 * 5 * 5
        combined_features_size = num_in_features + num_additional_features * 2
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_features_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self.additional_feature_layers = nn.Sequential(
            nn.Linear(num_additional_features, num_additional_features * 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(num_additional_features * 2, num_additional_features * 2),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x, additional_features):
        x = x.view(x.size(0), -1)
        
        processed_additional_features = self.additional_feature_layers(additional_features)
        combined_input = torch.cat((x, processed_additional_features), dim=1)
        return self.fc_layers(combined_input)
