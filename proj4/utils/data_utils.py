
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import FakeData

import matplotlib.pyplot as plt
import os
import sys
from attacks import fgsm
import random
from torch.utils.data import Subset

sys.path.append(os.path.abspath("../")) 

# Standard normalization transform for CIFAR-10 style data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_fake_data_loader():
    # Create fake dataset for testing purposes
    # Generates synthetic 32x32 RGB images with random labels
    fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=256, shuffle=False)
   
    return fakeloader, fakeset


class AdversarialAugmentedDataset(torch.utils.data.Dataset):
    # Dataset wrapper that generates adversarial examples on-the-fly
    # Each sample from base dataset is converted to adversarial example using FGSM
    
    def __init__(self, base_dataset, model, loss_fn, epsilon=2/255, budget=8/255, device='cuda'):
        self.base_dataset = base_dataset
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon  # Step size for FGSM
        self.budget = budget    # Maximum perturbation budget
        self.device = 'cuda'

    def __len__(self):
        # Return length of base dataset
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get original sample and convert to adversarial example
        x, y = self.base_dataset[idx]
        # Add batch dimension and move to device
        x = x.unsqueeze(0).to(self.device)
        y = torch.tensor(y).to(self.device)

        # Generate adversarial example using FGSM attack
        _, adv_x = fgsm(
                model=self.model,
                attacked_image_x=x.clone(),
                attacked_label_y=y,
                loss=self.loss_fn,
                targeted=False,  # Untargeted attack
                epsilon=self.epsilon,
                budget=self.budget,
                device=self.device
        )

        # Remove batch dimension and return adversarial sample
        return adv_x.squeeze(0).to(self.device), y.to(self.device)


class CombinedDataLoader:
    # Custom data loader that combines samples from two different dataloaders
    # Useful for mixing clean and adversarial examples in training
    
    def __init__(self, dataloader1, dataloader2, device):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.device = device

    def __iter__(self):
        # Iterate through both dataloaders simultaneously
        for data1, data2 in zip(self.dataloader1, self.dataloader2):
            x1, y1 = data1
            x2, y2 = data2
            
            # Move all tensors to specified device
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)

            # Concatenate batches from both dataloaders
            x_combined = torch.cat((x1, x2), dim=0)
            # Ensure labels have correct shape before concatenation
            y1 = y1.view(-1)
            y2 = y2.view(-1)
            y_combined = torch.cat((y1, y2), dim=0)

            yield x_combined, y_combined

    def __len__(self):
        # Length is determined by shorter of the two dataloaders
        return min(len(self.dataloader1), len(self.dataloader2))

def create_adv_loader(trainset, model, loss, epsilon, budget, device):
    # Create adversarial dataloader from subset of training data
    # Reduces computational cost by using only portion of original dataset
    
    # Create subset with 50% of original training data
    subset_size = max(1, int(0.5 * len(trainset)))
    subset_indices = random.sample(range(len(trainset)), subset_size)
    subset_trainset = Subset(trainset, subset_indices)

    # Create adversarial dataset wrapper
    adv_dataset = AdversarialAugmentedDataset(
        base_dataset=subset_trainset,
        model=model,
        loss_fn=loss,
        epsilon=epsilon,
        budget=budget,
        device=device
    )
    
    # Return dataloader with batch size 1 for adversarial generation
    return torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=True)