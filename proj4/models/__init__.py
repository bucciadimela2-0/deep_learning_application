import os
import torch
import sys
from torch import nn, optim
from .cnn import CNNclassic
from .autoencoder import Autoencoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proj1.models.cnn import CNN
from proj1.utils.data_loaders import get_dataloaders
from proj1.utils.train import train
from typing import Tuple
def get_model(name, device='cuda', load_pretrained=True, checkpoint_path=None, saved_dir="saved", input_shape: Tuple[int, ...] = None,
              num_classes: int = None,  **kwargs):
    """
    Get model - either load pretrained or train new one
    """
    name = name.lower()
    model = None
    
    # Handle loading of pre-trained models first 
    if load_pretrained and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
        model = load_model_from_checkpoint(checkpoint_path, name, device, input_shape, **kwargs)
        print(f"[INFO] Loaded pretrained model from {checkpoint_path}")
        return model


    # If no model or checkpoint, create and train the model
    if name == "cnnclassic":
        model = CNNclassic().to(device)
        loss = nn.CrossEntropyLoss()
        
    elif name == "autoencoder":
        model = Autoencoder().to(device)
        loss = nn.MSELoss()
        
    elif name == "cnn":
        channels, height, width = input_shape
        model = CNN().to(device)
        loss = nn.CrossEntropyLoss()
        
    else:
        raise ValueError(f"Unknown model name: {name}")

    # Setup optimizer and data
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Fix typo in dataset_name
    try:
        trainloader, _, _, _ = get_dataloaders(dataset_name="cifar10") 
    except Exception as e:
        print(f"Error loading data: {e}")
        return model  # Return untrained model if data loading fails
    
    # Training if no pre-trained model is found
    print(f"[INFO] Training {name} model...")
    model_trained = train(model, name, trainloader, optimizer, loss, device, saved=True, saved_dir=saved_dir)

    # Save the model after training
    if checkpoint_path is None:
        checkpoint_path = os.path.join(saved_dir, f"{name}.pt")
    
    os.makedirs(saved_dir, exist_ok=True)
    torch.save(model_trained.state_dict(), checkpoint_path)
    print(f"[INFO] Saved trained model to {checkpoint_path}")

    return model_trained


def load_model_from_checkpoint(checkpoint_path, model_name, device, input_shape, **kwargs):
    """
    Load model from checkpoint based on model name
    """
    # Load the state dict from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the model state dict correctly
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    #epoch = checkpoint.get('epoch', 'unknown')

    # Create the appropriate model based on name
    model_name = model_name.lower()
    if model_name == "cnnclassic":
        model = CNNclassic().to(device)
    elif model_name == "autoencoder":
        model = Autoencoder().to(device)
    elif model_name == "cnn":
        channels, height, width = input_shape
        model = CNN(input_channels=channels,
            input_size=input_shape,
            num_classes=10,
            num_filters=kwargs.get("num_filters", 64),
            num_blocks=kwargs.get("num_blocks", 1),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            use_batch_norm=kwargs.get("use_batch_norm", True),
            skip=kwargs.get("skip", False)).to(device)
    else:
        raise ValueError(f"Unknown model name for loading: {model_name}")
    
    #model_keys = set(model.state_dict().keys())
    #checkpoint_keys = set(state_dict.keys())
    model.load_state_dict(state_dict)
    return model

