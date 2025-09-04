import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import FakeData
import os
import random

def plot_score_distributions(scores_real, scores_fake, path):
    # Plot sorted score distributions for real vs fake data
    # Used to visualize separation between in-distribution and OOD samples
    
    plt.plot(sorted(scores_real.cpu()), label='test')
    plt.plot(sorted(scores_fake.cpu()), label='fake')
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_roc_curve(scores_real, scores_fake, path):
    # Generate ROC curve for OOD detection performance
    # Real data labeled as positive (1), fake data as negative (0)
    
    # Combine scores and create binary labels
    ypred = torch.cat((scores_real, scores_fake))
    y = torch.cat((torch.ones_like(scores_real), torch.zeros_like(scores_fake)))
    
    # Plot ROC curve using sklearn
    metrics.RocCurveDisplay.from_predictions(y.cpu(), ypred.cpu())
    plt.savefig(path)
    plt.close()

def confusion_matrix_plot(confusion_matrix, testset, path=None): 
    # Create normalized confusion matrix visualization
    # Shows classification performance as percentages
    
    # Convert to numpy and normalize by row (true class)
    confusion_matrix = confusion_matrix.cpu().numpy()
    cmn = confusion_matrix.astype(np.float32)
    cmn /= cmn.sum(axis=1, keepdims=True)  # Normalize to get percentages
    cmn_percent = (cmn * 100).astype(np.int32)
    
    # Create confusion matrix display
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cmn_percent, display_labels=testset.classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    
    # Customize plot appearance
    plt.title("Normalized confusion matrix (%)")
    plt.xlabel("Predicted class")
    plt.ylabel("Ground truth class")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{path}.png")
    plt.close()

    # Additional normalization (duplicate code - could be cleaned up)
    cmn = confusion_matrix.astype(np.float32)
    cmn /= cmn.sum(1)

def histogram_plot(model, testloader, testset, T=1, path=None):
    # Plot logit distribution for a random sample
    # Visualizes model's confidence distribution across classes
    
    # Get a batch of data
    for data in testloader:
        x, y = data

    # Select random sample from batch
    k = random.randint(0, x.shape[0] - 1)  # Fixed: was x.shape[0] which could cause index error
 
    # Get model logits for the sample
    output = model(x.cuda())
    
    # Create bar plot of logits across classes
    plt.bar(np.arange(10), output[k].detach().cpu())
    plt.title('logit')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()