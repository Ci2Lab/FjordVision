import torch
import torch.nn as nn

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, num_additional_features):
        super(BranchCNN, self).__init__()
        
        # Correct calculation of the combined input size
        combined_features_size = num_in_features + num_additional_features  # This should be 4096 + 3 = 4099
        
        self.fc_layers = nn.Sequential(
            # Adjust the first layer to take 4099 inputs
            nn.Linear(combined_features_size, 1024),  # Corrected to 4099
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Additional layers to process additional features more thoroughly
        self.additional_feature_layers = nn.Sequential(
            nn.Linear(num_additional_features, num_additional_features * 2),
            nn.ReLU(True),
            nn.Linear(num_additional_features * 2, num_additional_features),
            nn.ReLU(True),
        )

    def forward(self, x, additional_features):
        x = x.view(x.size(0), -1)
        
        # Process additional features to emphasize them
        processed_additional_features = self.additional_feature_layers(additional_features)
        
        # Combine the processed additional features with the main features
        combined_input = torch.cat((x, processed_additional_features), dim=1)
        return self.fc_layers(combined_input)