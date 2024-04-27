import torch.nn as nn
import torch.nn.functional as F


class Task1Network(nn.Module):
    def __init__(self):
        super(Task1Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.25)

        # Final convolution that maps directly to the number of output predictions
        self.final_conv = nn.Conv2d(
            128, 1, kernel_size=1
        )  # Output one channel with the same spatial dimensions

    def forward(self, x):
        # Apply convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Apply the final convolutional layer
        x = self.final_conv(x)

        # Sigmoid activation to get probabilities as output
        x = F.sigmoid(x)

        # If needed, reshape or adjust x to match exact output requirements
        return x
