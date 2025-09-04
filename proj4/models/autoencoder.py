import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from sklearn import metrics


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder: compresses input image to lower-dimensional representation
        self.encoder = nn.Sequential(
            # First conv layer: 32x32x3 -> 16x16x12
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            # Second conv layer: 16x16x12 -> 8x8x24
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            # Third conv layer: 8x8x24 -> 4x4x48
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            # Optional deeper layer (commented out for lighter model):
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),  # -> 2x2x96
            # nn.ReLU(),
        )
        
        # Decoder: reconstructs image from compressed representation
        self.decoder = nn.Sequential(
            # Optional deeper layer (commented out, matches encoder):
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # 2x2x96 -> 4x4x48
            # nn.ReLU(),
            # First transpose conv: 4x4x48 -> 8x8x24
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            # Second transpose conv: 8x8x24 -> 16x16x12
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            # Final transpose conv: 16x16x12 -> 32x32x3
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Ensure output values are in [0,1] range
        )

    def forward(self, x):
        # Encode input to compressed representation
        encoded = self.encoder(x)
        # Decode back to original image dimensions
        decoded = self.decoder(encoded)
        # Return both encoded representation and reconstructed image
        return encoded, decoded