import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from sklearn import metrics


class CNNclassic(nn.Module):
    def __init__(self):
        super().__init__()
        # Classic LeNet-style CNN architecture for CIFAR-10 classification
        
        # First convolutional layer: 3 input channels -> 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer: 2x2 pooling with stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: 6 input channels -> 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Fully connected layers for classification
        # Input size: 16 channels * 5x5 spatial dimensions = 400 features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Hidden layer: 120 -> 84 features
        self.fc2 = nn.Linear(120, 84)
        # Output layer: 84 -> 10 classes (CIFAR-10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Forward pass through the network
        
        # First conv block: conv -> relu -> maxpool
        # Input: 32x32x3 -> Conv: 28x28x6 -> Pool: 14x14x6
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block: conv -> relu -> maxpool
        # Input: 14x14x6 -> Conv: 10x10x16 -> Pool: 5x5x16
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        # Convert from 5x5x16 tensor to 400-dimensional vector
        x = torch.flatten(x, 1)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Output layer (no activation - raw logits for classification)
        x = self.fc3(x)
        
        return x
    