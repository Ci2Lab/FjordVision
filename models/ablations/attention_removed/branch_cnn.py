import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes, dropout_rate=0.5):
        super(BranchCNN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(num_in_features, 4096),
            nn.BatchNorm1d(4096),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, additional_features):
        combined_input = torch.cat((x, additional_features), dim=1)
        return self.fc_layers(combined_input)
