import torch
import torch.nn.functional as F
import numpy as np
from utils import load_DCT_basis_torch

class Djpegnet(torch.nn.Module):
    def __init__(self, device):
        super(Djpegnet, self).__init__()
        self.dct_basis = load_DCT_basis_torch().float()
        self.dct_basis = self.dct_basis.to(device)

        self.conv1a = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.bn1a = torch.nn.BatchNorm2d(num_features=64)
        self.conv1b = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.bn1b = torch.nn.BatchNorm2d(num_features=64)
        self.mp1b = torch.nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.conv2a = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=1, padding=2)
        self.bn2a = torch.nn.BatchNorm2d(num_features=128)
        self.mp2a = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=1, padding=2)
        self.bn3a = torch.nn.BatchNorm2d(num_features=256)
        self.mp3a = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(15*8*256+64, 500)
        self.fc2 = torch.nn.Linear(564, 500)
        self.fc3 = torch.nn.Linear(564, 2)

    def forward(self, x, qvectors):
        # feature extraction
        with torch.no_grad():
            gamma=1e+06
            x = F.conv2d(x, self.dct_basis, stride=8)
            for b in range(-60, 61):
                x_ = torch.sum(torch.sigmoid(gamma*(x-b)), axis=[2,3])/1024
                x_ = torch.unsqueeze(x_, axis=1)
                if b==-60:
                    features = x_
                else:
                    features = torch.cat([features, x_], axis=1)
            features = features[:, 0:120, :] - features[:, 1:121, :]
            features = torch.reshape(features, (-1, 1, 120, 64))

        # convolutional layers
        x = F.relu(self.bn1a(self.conv1a(features)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.mp1b(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.mp2a(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.mp3a(x)

        x_flat = torch.reshape(x, (-1, 15*8*256))

        # fully connected layers
        x = torch.cat([qvectors, x_flat], axis=1)
        x = F.relu(self.fc1(x))
        x = torch.cat([qvectors, x], axis=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([qvectors, x], axis=1)
        x = self.fc3(x)

        return x


    

