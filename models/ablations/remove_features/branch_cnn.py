import torch.nn as nn

class BranchCNN(nn.Module):
    def __init__(self, in_channels, num_classes, output_size):
        super(BranchCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.Mish()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * output_size[0] * output_size[1], 512),
            nn.Mish(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
