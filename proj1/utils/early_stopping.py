import os
import torch
import numpy as np
import random
import time

from torch import optim

class EarlyStopping:
   
   #Early stopping to prevent overfitting during training.
   
   def __init__(self, patience: int = 5, delta: float = 0):
       self.patience = patience
       self.delta = delta
       self.best_score = None
       self.early_stop = False
       self.counter = 0

   def __call__(self, val_loss: float):
    
       #Check if training should stop based on validation loss.
      
       # Convert to "higher is better" for simpler logic
       score = -val_loss
       
       if self.best_score is None:
           # First epoch - initialize best score
           self.best_score = score
           self.counter = 0
       elif score < self.best_score + self.delta:
           # No significant improvement
           self.counter += 1
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           # Significant improvement found
           self.best_score = score
           self.counter = 0  # Reset counter

   def reset(self):
       #Reset early stopping state for new training run.
       self.best_score = None
       self.early_stop = False
       self.counter = 0