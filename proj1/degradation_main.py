import torch
from torch import nn
from torch.optim import Adam
import logging
import os
from datetime import datetime

from utils.train import train_one_epoch, validate
from utils.config_utils import parse_config, print_config_summary, get_model_name, validate_config
from utils import (
    get_dataloaders, get_dataset_info, save_checkpoint, load_model,
    set_seed, EarlyStopping, init_wandb, log_metrics, log_images, watch_model
)
from models import get_model


def setup_logger(config):
    # Setup logging configuration for training
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp and model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.get("params", {}).get("name", config.model["name"])
    log_filename = f"{log_dir}/{model_name}_{timestamp}.log"
    
    # Configure logging with file handler only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler()  # Also print to console
        ]
    )
    
    # Create and return logger instance
    logger = logging.getLogger(__name__)
    return logger


def main():
    # Parse and validate configuration from command line and YAML file
    config, args = parse_config()
    validate_config(config)
    
    # Setup logger for this training session
    logger = setup_logger(config)
    
    # Setup device and ensure reproducibility
    set_seed(config.seed)  # Set random seeds for reproducible results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare datasets
    train_loader, val_loader, trainset, valset = get_dataloaders(
        dataset_name=config.dataset["name"],
        augment=config.dataset["augment"],  # Apply data augmentation if specified
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    # Get dataset metadata (channels, height, width, classes)
    info = get_dataset_info(config.dataset["name"])

    # Initialize model with dataset-specific parameters
    model = get_model(
        config.model["name"],
        input_shape=(info["channels"], info["height"], info["width"]),
        num_classes=info["classes"], 
        **config.model.get("params", {})  # Pass additional model parameters
    ).to(device)  # Move model to GPU/CPU

    # Resume from checkpoint if specified in config
    if config.resume:
        model = load_model(config.resume, model, device)

    # Setup training components
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = Adam(model.parameters(), lr=config.lr)  # Adam optimizer
    early_stopper = EarlyStopping(patience=config.patience, delta=config.delta)  # Early stopping to prevent overfitting

    # Initialize experiment tracking with Weights & Biases if enabled
    if config.wandb["enabled"]:
        run = init_wandb(config)  # Initialize W&B run
        watch_model(model)  # Track model gradients and parameters

    # Initialize tracking variables for best metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Main training loop
    for epoch in range(1, config.epochs + 1):
        # Training phase: update model weights
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase: evaluate model without updating weights
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Track best metrics achieved so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Log metrics to W&B if enabled
        if config.wandb["enabled"]:
            log_metrics({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

        # Check for early stopping based on validation loss
        early_stopper(val_loss)
        if early_stopper.early_stop:
            break  # Stop training if no improvement for specified patience epochs

    # Save final model if specified in configuration
    if config.save_model:
        model_path = f"{config.save_dir}/{get_model_name(config)}.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if needed
        save_checkpoint(model, optimizer, epoch, path=model_path)  # Save model state


if __name__ == "__main__":
    main()