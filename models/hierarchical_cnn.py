import torch
import torch.nn as nn
from .branch_cnn import BranchCNN

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes_hierarchy, num_additional_features):
        super(HierarchicalCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])

        # Branches for each convolutional layer output
        self.branches = nn.ModuleList([
            BranchCNN(128 * 64 * 64, num_classes_hierarchy[0], num_additional_features),  # Update sizes accordingly
            BranchCNN(256 * 32 * 32, num_classes_hierarchy[1], num_additional_features),
            BranchCNN(512 * 16 * 16, num_classes_hierarchy[2], num_additional_features),
            BranchCNN(512 * 8 * 8, num_classes_hierarchy[3], num_additional_features)
        ])

    def forward(self, x, conf, iou, pred_species):
        outputs = []
        additional_features = torch.cat((conf.view(-1, 1), iou.view(-1, 1), pred_species.view(-1, 1)), dim=1)

        for conv_layer, branch in zip(self.conv_layers, self.branches):
            x = conv_layer(x)
            branch_output = branch(x, additional_features)
            outputs.append(branch_output)

        return outputs