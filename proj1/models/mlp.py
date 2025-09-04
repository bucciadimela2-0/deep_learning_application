import torch
from torch import nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):

   def __init__(self,  
               layer_sizes: List[int], 
               activation: str = "relu",
               dropout_rate: float = 0.0,
               batch_norm: bool = False,
               name: str = "mlp_base"
              ):
       super(MLP, self).__init__()
   
       # Validate input: need at least input and output layer
       if len(layer_sizes) < 2:
           raise ValueError("layer_sizes must contain at least 2 elements (input and output)")
       
       # Store configuration
       self.layer_sizes = layer_sizes
       self.activation_name = activation
       self.name = name
       self.num_layers = len(layer_sizes) - 1  # Number of linear transformations
       self.dropout_rate = dropout_rate
       self.batch_norm = batch_norm
       
       # Build network layers
       self.layers = nn.ModuleList()
       self.batch_norms = nn.ModuleList() if batch_norm else None
       self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
       
       # Create each linear layer with optional batch norm and dropout
       for i in range(self.num_layers):
           linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
           self.layers.append(linear)
           
           # Add batch normalization (skip output layer)
           if batch_norm and i < self.num_layers - 1:
               self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))
           
           # Add dropout (skip output layer)
           if dropout_rate > 0 and i < self.num_layers - 1:
               self.dropouts.append(nn.Dropout(dropout_rate))

       # Set activation function
       self.activation = self._get_activation_function(activation)
   
   def _get_activation_function(self, activation: str):
       #Get activation function from string name
       activations = {
           "tanh": torch.tanh,
           "sigmoid": torch.sigmoid,
           "relu": F.relu,
           "leaky_relu": lambda x: F.leaky_relu(x, 0.01)
       }
       if activation not in activations:
           raise ValueError(f"Activation {activation} not supported. Use: {list(activations.keys())}")
       return activations[activation]

   def forward(self, x):

       # Flatten input for fully connected layers
       x = x.view(x.size(0), -1)
       
       # Pass through each layer
       for i, layer in enumerate(self.layers):
           x = layer(x)
           
           # Apply activation, batch norm, dropout to hidden layers only
           if i < self.num_layers - 1:  # Skip output layer
               x = self.activation(x)
               
               if self.batch_norm:
                   x = self.batch_norms[i](x)
               
               if self.dropout_rate > 0:
                   x = self.dropouts[i](x)
       
       return x  