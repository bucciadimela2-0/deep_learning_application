# main.py
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from utils.data_utils import create_adv_loader, CombinedDataLoader

from utils.ood_eval import evaluate_ood
from models import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proj1.utils.train import *


def load_config(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_device(config):
    # Setup computation device (GPU/CPU) based on configuration
    device_name = config.get('device', 'auto')
    
    if device_name == 'auto':
        # Automatically select GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # Use specified device
        device = torch.device(device_name)
    
    return device

def get_model(model_config, device):
    # Create model instance based on configuration
    model_type = model_config['name']
    
    if model_type == 'cnn':
        # Load CNN classifier model
        from models.cnn import CNNclassic
        model = CNNclassic()
    elif model_type == 'autoencoder':
        # Load autoencoder model for reconstruction tasks
        from models.autoencoder import Autoencoder
        model = Autoencoder()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to specified device
    return model.to(device)

def get_optimizer(model, optimizer_config):
    # Create optimizer based on configuration
    opt_type = optimizer_config['type']
    lr = optimizer_config['lr']
    
    if opt_type == 'adam':
        # Adam optimizer with adaptive learning rates
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == 'sgd':
        # SGD with momentum
        return torch.optim.SGD(model.parameters(), lr=lr, 
                              momentum=optimizer_config.get('momentum', 0.9))
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

def get_criterion(loss_config):
    # Create loss function based on configuration
    loss_type = loss_config['type']
    
    if loss_type == 'crossentropy':
        # CrossEntropy for classification tasks
        return nn.CrossEntropyLoss()
    elif loss_type == 'mse':
        # MSE for reconstruction tasks (autoencoders)
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main():
    # Main training pipeline with YAML configuration
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train models with YAML config")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load and setup configuration
    config = load_config(args.config)
    
    # Setup computation device
    device = setup_device(config)
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['saved_dir'], exist_ok=True)
    
    # Initialize model components
    model = get_model(config['model'], device)
    optimizer = get_optimizer(model, config['optimizer'])
    criterion = get_criterion(config['loss'])
    is_autoencoder = config['model']['name'] == 'autoencoder'
    
    # Setup data loaders with appropriate augmentation
    augment = 'zero' if is_autoencoder else 'none'  # Zero normalization for autoencoders
    trainloader, valloader, trainset, testset = get_dataloaders(config['data']['dataset'], augment, augment)
    
    # Training configuration
    train_config = config['training']
    model_name = config['model']['name']
    
    # Setup adversarial training if enabled
    if train_config.get('adversarial', {}).get('enabled', False):
        # Create adversarial examples loader
        adv_config = train_config['adversarial']
        
        adv_loader = create_adv_loader(
            trainset=trainset,
            model=model,
            loss=criterion,
            epsilon=adv_config['epsilon'],  # Perturbation strength
            budget=adv_config['budget'],    # Maximum perturbation budget
            device=device
        )
        
        # Combine clean and adversarial examples
        final_trainloader = CombinedDataLoader(trainloader, adv_loader, device)
        model_name += "_adversarial"
    else:
        # Use standard training data only
        final_trainloader = trainloader

    # Execute training process
    trained_model = train(
        model=model,
        model_name=model_name,
        trainloader=final_trainloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=train_config['epochs'],
        saved=train_config['save_model'],
        saved_dir=config['paths']['saved_dir'],
        is_autoencoder=is_autoencoder  # Flag for autoencoder-specific training logic
    )
    
    # Run OOD evaluation if enabled
    if config.get('evaluation', {}).get('enabled', False):
        eval_config = config['evaluation']
        
        # Evaluate model's out-of-distribution detection capabilities
        evaluate_ood(
            model=trained_model,
            model_type=eval_config['model_type'],
            device=device,
            plot_prefix=eval_config.get('plot_prefix', model_name)
        )


if __name__ == "__main__":
    main()