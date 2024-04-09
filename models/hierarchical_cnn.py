import torch
import torch.nn as nn
from .branch_cnn import BranchCNN

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features, output_size=(5, 5)):
        super(HierarchicalCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=5, padding=2),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(30),
        )
        
        # Using dense connections, concatenating features from previous layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=5, padding=2),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(60),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(60, 100, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(100),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 150, kernel_size=3, padding=1),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(150),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)

        self.branches = nn.ModuleList([
            BranchCNN(output_size[0] * output_size[1] * 150, num_classes, num_additional_features)
            for num_classes in num_classes_hierarchy
        ])

    def forward(self, x, conf, iou, pred_species):
        outputs = []
        additional_features = torch.cat((conf.view(-1, 1), iou.view(-1, 1), pred_species.view(-1, 1)), dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        for branch in self.branches:
            branch_output = branch(x, additional_features)
            outputs.append(branch_output)

        return outputs