import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfDrivingModel(nn.Module):
    def __init__(self):
        super(SelfDrivingModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(
            in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = nn.Conv2d(
            in_channels=48, out_channels=64, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, image):
        x = F.elu(self.conv1(image))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        x = x.view(x.shape[0], -1)

        x = F.elu(self.fc1(x))
        x = self.dropout1(x)
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.dropout2(x)
        x = F.elu(self.fc4(x))

        return x
