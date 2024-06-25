import torch.nn as nn

class BranchCNN(nn.Module):
    def __init__(self, num_in_features, num_classes):
        super(BranchCNN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(num_in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(x)
