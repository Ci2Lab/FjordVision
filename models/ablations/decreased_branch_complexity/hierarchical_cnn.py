import torch
import torch.nn as nn
from .branch_cnn import BranchCNN

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            Mish(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return nn.Sigmoid()(out).view(x.size(0), x.size(1), 1, 1) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(combined))
        return x * attention.expand_as(x)

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features, dropout_rate=0.5):
        super(HierarchicalCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ChannelAttention(64),
            SpatialAttention(kernel_size=5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            Mish(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ChannelAttention(128),
            SpatialAttention(kernel_size=5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            Mish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ChannelAttention(256),
            SpatialAttention(kernel_size=5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            Mish(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ChannelAttention(512),
            SpatialAttention(kernel_size=5),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))

        # Creating branch CNNs for each hierarchical level
        self.binary_branch = BranchCNN(64, num_classes_hierarchy[0], dropout_rate)
        self.class_branch = BranchCNN(128, num_classes_hierarchy[1], dropout_rate)
        self.genus_branch = BranchCNN(256, num_classes_hierarchy[2], dropout_rate)
        self.species_branch = BranchCNN(512, num_classes_hierarchy[3], dropout_rate)

    def forward(self, x, conf, pred_species):
        additional_features = torch.cat((conf.view(-1, 1), pred_species.view(-1, 1)), dim=1)

        x1 = self.conv1(x)
        x1_pooled = self.global_avg_pool(x1).view(x1.size(0), -1)
        binary_output = self.binary_branch(x1_pooled, additional_features)

        x2 = self.conv2(x1)
        x2_pooled = self.global_avg_pool(x2).view(x2.size(0), -1)
        class_output = self.class_branch(x2_pooled, additional_features)

        x3 = self.conv3(x2)
        x3_pooled = self.global_avg_pool(x3).view(x3.size(0), -1)
        genus_output = self.genus_branch(x3_pooled, additional_features)

        x4 = self.conv4(x3)
        x4_pooled = self.global_avg_pool(x4).view(x4.size(0), -1)
        species_output = self.species_branch(x4_pooled, additional_features)

        return [binary_output, class_output, genus_output, species_output]
