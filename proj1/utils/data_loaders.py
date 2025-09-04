import sys
import os
import random

import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# Add parent directory to path for imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def get_transform(dataset_name="mnist", augment_type="none"):
    # Define mean and standard deviation for normalization based on dataset
    if dataset_name == "mnist":
        mean, std = [0.1307], [0.3081]
    elif dataset_name == "cifar10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
        raise ValueError("Unsupported dataset")

    # Base transformations applied to all images
    base = [
        v2.ToImage(),  # Convert PIL image to tensor
        v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0,1]
        v2.Normalize(mean=mean, std=std)  # Normalize with dataset-specific statistics
    ]

    # No augmentation - return base transforms only
    if augment_type == "none":
        return v2.Compose(base)

    # Simple augmentation with basic geometric transforms
    elif augment_type == "simple":
        transforms = [
            v2.RandomRotation(10),  # Random rotation up to 10 degrees
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Small translation and scaling
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),  # Gaussian blur with 30% probability
        ] + base

    # Heavy augmentation with aggressive transforms
    elif augment_type == "heavy":
        transforms = [
            v2.RandomRotation(25),  # Random rotation up to 25 degrees
            v2.RandomAffine(0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),  # Large translation, scaling, and shear
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective distortion
            v2.ElasticTransform(alpha=50.0, sigma=5.0),  # Elastic deformation
            v2.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),  # Strong Gaussian blur
            v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),  # Random erasing patches
        ] + base
    
    # Zero-centered normalization (transforms to [-1, 1] range)
    elif augment_type == "zero":
        mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)  # Normalize to [-1, 1] range
        ]

    else:
        raise ValueError(f"Unknown augment_type: {augment_type}")

    return v2.Compose(transforms)


class CustomDataset(Dataset):
    def __init__(self, dataset_name="mnist", train=True, augment="none"):
        # Store dataset configuration
        self.dataset_name = dataset_name
        self.train = train
        self.augment = augment

        # Load MNIST dataset
        if dataset_name == "mnist":
            dataset = datasets.MNIST(root="./data", train=train, download=True)
            # Convert tensor data to PIL Images for consistency
            self.data = []
            for img in dataset.data:
                img_pil = PILImage.fromarray(img.numpy(), mode='L')  # Convert to grayscale PIL image
                self.data.append(img_pil)
            self.classes = dataset.classes
            
        # Load CIFAR10 dataset
        elif dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root="./data", train=train, download=True)
            # Convert numpy arrays to PIL Images
            self.data = []
            for img in dataset.data:
                img_pil = PILImage.fromarray(img, mode='RGB')  # Convert to RGB PIL image
                self.data.append(img_pil)
            self.classes = dataset.classes
        else:
            raise ValueError("Unsupported dataset")

        # Extract labels (handle different attribute names)
        self.labels = dataset.targets if hasattr(dataset, 'targets') else dataset._labels
        # Get appropriate transforms for this dataset and augmentation type
        self.transforms = get_transform(dataset_name, augment)

    def __getitem__(self, idx):
        # Get image and label at specified index
        image, label = self.data[idx], self.labels[idx]
        # Apply transforms to the image
        image = self.transforms(image)
        return image, label

    def __len__(self):
        # Return total number of samples
        return len(self.labels)


def get_dataloaders(dataset_name="mnist", augment="none", augment_test="none", batch_size=64, num_workers=2):
    # Create training dataset with augmentation
    trainset = CustomDataset(dataset_name, train=True, augment=augment)
    # Create test dataset with separate augmentation (usually none)
    testset = CustomDataset(dataset_name, train=False, augment=augment_test)

    # Create data loaders for training and testing
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, trainset, testset


def plot_batch_images(images, save_path, dataset_name="mnist", title="Sample Batch"):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert grayscale to RGB for consistent plotting
    if dataset_name == "mnist" and images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)  # Repeat grayscale channel to create RGB
    
    # Create grid of images
    grid = make_grid(images, nrow=8, padding=3, normalize=True, pad_value=0.8)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(grid.permute(1, 2, 0).clamp(0, 1))  # Permute dimensions for matplotlib and clamp values
    ax.set_title(f"{dataset_name.upper()} - {title}", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')  # Remove axes
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # Close to free memory


def get_dataset_info(dataset_name):
    # Return dataset-specific information
    if dataset_name == "mnist":
        return {
            'channels': 1,  # Grayscale
            'height': 28,
            'width': 28,
            'classes': 10,  # Digits 0-9
            'mean': [0.1307],  # Dataset mean for normalization
            'std': [0.3081]   # Dataset std for normalization
        }
    elif dataset_name == "cifar10":
        return {
            'channels': 3,  # RGB
            'height': 32,
            'width': 32,
            'classes': 10,  # 10 object classes
            'mean': [0.4914, 0.4822, 0.4465],  # Per-channel means
            'std': [0.2023, 0.1994, 0.2010]    # Per-channel standard deviations
        }


def denormalize_and_save_image(tensor, path, title=""):
    # Denormalize an image tensor and save it as an image file
    
    # Inverse normalization (assuming normalization with [0.5, 0.5, 0.5] mean and std)
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[2.0, 2.0, 2.0])  # Reverse the normalization
    ])
    
    # Apply denormalization
    tensor = inv_transform(tensor)

    # Convert tensor to numpy array for matplotlib
    tensor = tensor.squeeze().cpu().detach().numpy()
    
    # Save image
    plt.imshow(tensor.transpose(1, 2, 0))  # Transpose from CHW to HWC format
    plt.title(title)
    plt.axis('off')  # Remove axes
    plt.savefig(path, bbox_inches='tight')
    plt.close()  # Close to free memory