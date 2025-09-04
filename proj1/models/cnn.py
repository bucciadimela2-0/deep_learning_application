import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, skip=False):
        super().__init__()
        padding = kernel_size // 2
        self.skip = skip and in_channels == out_channels  # Skip connection only if same dimensions

        # Build convolutional block
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip:
            out = out + identity  # Add skip connection
        return self.relu(out)


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,          
        num_blocks: int = 1,            
        num_filters: int = 64,         
        input_size: Tuple[int, int] = (32, 32),  
        input_channels: int = 3,        
        use_batch_norm: bool = True,    
        dropout_rate: float = 0.5,      
        skip: bool = False,             
    ):
        super().__init__()

        # Initial convolution to adapt input
        self.input_adapter = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters) if use_batch_norm else nn.Identity(),
            nn.ReLU()
        )

        # Two main layers with different filter counts
        self.layer1 = self._make_layer(num_filters, num_filters, num_blocks, use_batch_norm, skip)
        self.layer2 = self._make_layer(num_filters, num_filters * 2, num_blocks, use_batch_norm, skip)

        # Final classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_filters * 2, num_classes)
        
        # Grad-CAM variables
        self.gradients = None
        self.activations = None
        self.hooks = []
    #Build a layer with multiple blocks
    def _make_layer(self, in_channels, out_channels, num_blocks, use_batch_norm, skip):
        
        layers = []
        # First block handles channel dimension change
        layers.append(BasicBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, skip=skip))
        # Remaining blocks maintain same dimensions
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, skip=skip))
        return nn.Sequential(*layers)
    #Register hooks for Grad-CAM on last convolutional layer
    def register_gradient_hooks(self):
       
        last_conv = None
        
        # Find the last conv layer in layer2
        for name, module in self.layer2.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                block_name = name
        
        if last_conv is None:
            raise ValueError("No convolutional layers found in layer2")
        
        print(f"Using layer2.{block_name} for Grad-CAM")
        
        # Hook to save activations (forward pass)
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        # Hook to save gradients (backward pass)
        def save_gradient(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Register the hooks
        handle1 = last_conv.register_forward_hook(save_activation)
        handle2 = last_conv.register_full_backward_hook(save_gradient)
        
        self.hooks = [handle1, handle2]
    #Remove all registered hooks 
    def remove_hooks(self):
       
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None

    #Calculate Grad-CAM attention map
    def get_gradcam(self):
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not available. Call register_gradient_hooks() first.")
        
        # Average gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Keep only positive contributions
        cam = F.relu(cam)
        
        return cam
    #Forward pass 
    def forward(self, x, return_cam=False):
       
        x = self.input_adapter(x)
        x = self.layer1(x)
        
        # layer2 is where we capture activations for Grad-CAM
        x = self.layer2(x)
        
        # Continue with rest of the network
        x = self.avgpool(x)              
        x = torch.flatten(x, 1)        
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits