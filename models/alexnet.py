import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(48, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(48, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(256, 192, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 192, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(),
        )

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        o1_1 = self.conv1_1(x)
        o1_2 = self.conv1_2(x)
        o2_1 = self.conv2_1(o1_1)
        o2_2 = self.conv2_2(o1_2)
        o2 = torch.cat((o2_1, o2_2), dim=1)
        o3_1 = self.conv3_1(o2)
        o3_2 = self.conv3_2(o2)
        o4_1 = self.conv4_1(o3_1)
        o4_2 = self.conv4_2(o3_2)
        o5_1 = self.conv5_1(o4_1)
        o5_2 = self.conv5_2(o4_2)

        return torch.cat((o5_1, o5_2), dim=1)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = ConvLayer()
        # self.average_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        return self.classifier(feature)
