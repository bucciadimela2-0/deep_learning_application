
###Model analysis utilities for Grad-CAM and adversarial attacks.

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging


class ModelAnalyzer:
    #Model analysis class for Grad-CAM and FGSM attacks
    
    def __init__(self, model, device='cuda', logger=None):
        # Initialize the model analyzer with model, device, and logger
        self.model = model
        self.device = device
        self.model.to(device)  # Move model to specified device
        self.logger = logger or logging.getLogger('ModelAnalyzer')  # Use provided logger or create default
        # CIFAR-10 class names for interpretation
        self.cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                             'dog', 'frog', 'horse', 'ship', 'truck']

    def get_prediction(self, image):
        #Get model prediction and confidence
        # Disable gradient computation for inference
        with torch.no_grad():
            # Forward pass through the model
            output = self.model(image.unsqueeze(0))  # Add batch dimension
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(output, dim=1)
            # Get the highest confidence prediction
            confidence, pred = torch.max(probs, 1)
        return pred.item(), confidence.item()
    
    def apply_gradcam(self, image, target_class=None):
        #Apply Grad-CAM to visualize model attention
        # Set model to evaluation mode
        self.model.eval()
        # Clear any existing gradients
        self.model.zero_grad()
        # Register hooks to capture gradients and feature maps
        self.model.register_gradient_hooks()
        
        # Forward pass with CAM enabled
        output = self.model(image.unsqueeze(0), return_cam=True)  # Add batch dimension
        # Use predicted class if no target specified
        if target_class is None:
            target_class = output.argmax().item()
        
        # Get the target class score and compute gradients
        target = output[0, target_class]
        target.backward()  # Backpropagate to get gradients
        
        # Extract and normalize the CAM
        cam = self.model.get_gradcam().squeeze().detach().cpu().numpy()
        # Normalize CAM values to [0, 1] range
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Add epsilon to avoid division by zero
        
        # Clean up hooks to prevent memory leaks
        self.model.remove_hooks()
        return cam, target_class

    def fgsm_attack(self, image, target_class, epsilon=0.03):
        #Apply FGSM adversarial attack
        # Clone image and enable gradient computation
        image_adv = image.clone().detach().requires_grad_(True)
        # Forward pass to get model output
        output = self.model(image_adv.unsqueeze(0))  # Add batch dimension
        # Compute loss with respect to target class
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([target_class]).to(self.device))
        
        # Clear gradients and compute loss gradients
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial perturbation (targeted attack uses negative gradient)
        perturbation = -epsilon * image_adv.grad.sign()
        # Apply perturbation and clamp to valid pixel range [0, 1]
        return torch.clamp(image_adv + perturbation, 0, 1).detach()

    def visualize_gradcam(self, image, cam, title="Grad-CAM", save_path=None, alpha=0.6):
        #Visualize Grad-CAM overlay
        # Convert tensor to numpy array if necessary
        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().numpy()
            # Convert from CHW to HWC format if needed
            if img_np.shape[0] == 3:  # Check if channels-first format
                img_np = img_np.transpose(1, 2, 0)
        
        # Normalize image to [0, 1] range for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Resize CAM to match image dimensions
        cam_resized = torch.nn.functional.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=img_np.shape[:2], mode='bilinear'  # Resize to image height and width
        ).squeeze().numpy()
        
        # Create colored heatmap using jet colormap
        heatmap = plt.get_cmap('jet')(cam_resized)[:, :, :3]  # Take only RGB channels
        # Blend heatmap with original image
        superimposed = alpha * heatmap + (1-alpha) * img_np
        
        # Create visualization with original and CAM overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Show original image
        ax1.imshow(img_np)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Show Grad-CAM overlay
        ax2.imshow(superimposed)
        ax2.set_title(title)
        ax2.axis('off')
        
        # Save visualization if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            self.logger.info(f"Saved Grad-CAM visualization to {save_path}")
        
        plt.close()  # Close figure to free memory

    def visualize_attack(self, original, attacked, epsilon, save_path=None):
        #Visualize FGSM attack results
        # Calculate absolute difference as perturbation visualization
        perturbation = (attacked - original).abs()
        
        # Prepare images and titles for visualization
        images = [original, attacked, perturbation]
        titles = ['Original', f'FGSM Attack (ε={epsilon})', 'Perturbation']
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (img, title) in enumerate(zip(images, titles)):
            # Convert tensor to numpy if necessary
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
                # Convert from CHW to HWC format if needed
                if img_np.shape[0] == 3:
                    img_np = img_np.transpose(1, 2, 0)
                # Normalize for visualization
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Display image with title
            axes[i].imshow(img_np)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            self.logger.info(f"Saved attack visualization to {save_path}")
        
        plt.close()  # Close figure to free memory