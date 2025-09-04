"""
Utility functions for model analysis.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os
import logging
from datetime import datetime

# Import existing dataset utilities
from .data_loaders import get_dataloaders


def log_analysis_config(config):
    # Setup logging configuration for analysis
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp and model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.get("params", {}).get("name", config.model["name"])
    log_filename = f"{log_dir}/gradcam_{model_name}_{timestamp}.log"
    
    # Configure logging with file handler only (no console output)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler()  # Also print to console if needed
        ]
    )
    
    # Create logger instance and log initial info
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_filename}")
    return logger


def get_test_dataset(dataset_name, ood=False, logger=None):
    # Get test dataset for analysis using existing utilities
    
    if ood or dataset_name == 'mnist_ood':
        # For OOD testing: use MNIST on CIFAR-trained models
        if logger:
            logger.info("Using MNIST dataset for OOD testing on CIFAR model")
        
        # Use existing get_dataloaders but modify for OOD
        _, test_loader, _, _ = get_dataloaders(
            dataset_name="mnist", 
            augment="none", 
            batch_size=1, 
            num_workers=1
        )
        
        # Need to adapt MNIST to CIFAR input format (1->3 channels, 28->32 size)
        class OODDatasetWrapper:
            def __init__(self, original_loader):
                self.original_loader = original_loader
                # Transform to resize and convert channels for compatibility
                self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),  # Resize from 28x28 to 32x32
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Convert grayscale to RGB
                ])
                
            def __iter__(self):
                for batch in self.original_loader:
                    images, labels = batch
                    # Transform each image in the batch
                    transformed_images = torch.stack([
                        self.transform(img) for img in images
                    ])
                    yield transformed_images, labels
        
        return OODDatasetWrapper(test_loader)
    else:
        # Normal dataset - use existing utilities
        if logger:
            logger.info(f"Using {dataset_name.upper()} dataset")
        
        # Get standard test loader with no augmentation
        _, test_loader, _, _ = get_dataloaders(
            dataset_name=dataset_name,
            augment="none", 
            batch_size=1, 
            num_workers=1
        )
        return test_loader


def load_checkpoint(model, checkpoint_path, device, logger):
    # Load model checkpoint with compatibility checking
    
    try:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        # Load checkpoint to specified device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Format with metadata (epoch, optimizer state, etc.)
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            logger.info(f"Loading from model_state_dict (epoch: {epoch})")
        else:
            # Direct state_dict format (just model parameters)
            state_dict = checkpoint
            logger.info("Loading direct state_dict")
        
        # Log key information for debugging
        logger.info(f"State dict keys (first 5): {list(state_dict.keys())[:5]}...")
        logger.info(f"Model keys (first 5): {list(model.state_dict().keys())[:5]}...")
        
        # Check compatibility between model and checkpoint
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Find missing and unexpected keys
        missing_keys = model_keys - checkpoint_keys  # Keys in model but not in checkpoint
        unexpected_keys = checkpoint_keys - model_keys  # Keys in checkpoint but not in model
        
        # Report compatibility issues
        if missing_keys or unexpected_keys:
            logger.error("Architecture mismatch detected!")
            logger.error(f"Missing keys: {len(missing_keys)} (first 5: {list(missing_keys)[:5]})")
            logger.error(f"Unexpected keys: {len(unexpected_keys)} (first 5: {list(unexpected_keys)[:5]})")
            logger.error("Suggestions:")
            logger.error("1. Check model configuration matches training config")
            logger.error("2. Verify model_name and model_params are correct")
            return False
        
        # Load model state if compatible
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded model from {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return False


def log_model_info(model, config, logger):
    # Log model information including name and parameter count
    
    # Get model name from model attribute or config
    model_name = model.name if hasattr(model, 'name') else config.model['name']
    # Count total number of parameters
    param_count = sum(p.numel() for p in model.parameters())