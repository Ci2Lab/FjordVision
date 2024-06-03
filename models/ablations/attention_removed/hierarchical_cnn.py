import torch
import torch.nn as nn
from .branch_cnn import BranchCNN

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features, output_size=(5, 5)):
        super(HierarchicalCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)

        # Initialize BranchCNN with the correct size and additional features
        self.branches = nn.ModuleList([
            BranchCNN(output_size[0] * output_size[1] * 256, num_classes, num_additional_features)
            for num_classes in num_classes_hierarchy
        ])

    def forward(self, x, conf, pred_species):
        additional_features = torch.cat((conf.view(-1, 1), pred_species.view(-1, 1)), dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_avg_pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output before passing to the branches

        outputs = []
        for branch in self.branches:
            branch_output = branch(x, additional_features)
            outputs.append(branch_output)

        return outputs
