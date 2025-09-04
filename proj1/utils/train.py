import torch
import os
from .data_loaders import get_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device, is_autoencoder=False):
    # Train the model for one epoch
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct, total = 0, 0

    # Process each batch in the dataloader
    for images, labels in dataloader:
        # Move data to specified device
        images, labels = images.to(device), labels.to(device)
        
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        
        if is_autoencoder:
            # For autoencoders: outputs is (encoded, decoded)
            if isinstance(outputs, tuple):
                encoded, decoded = outputs
                # Calculate reconstruction loss (MSE between original and reconstructed)
                loss = criterion(images, decoded)
            
        else:
            # For normal classifiers
            loss = criterion(outputs, labels)
            # Calculate predictions and accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
        
        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Track running statistics
        running_loss += loss.item()
        total += labels.size(0)
        
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    # Return average loss and accuracy
    if is_autoencoder:
        return running_loss /len(dataloader),  correct / total  # Accuracy = 0 for autoencoder
    else:
        return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device, is_autoencoder=False):
    # Validate the model without updating weights
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct, total = 0, 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            if is_autoencoder:
                # For autoencoders: calculate reconstruction loss
                if isinstance(outputs, tuple):
                    encoded, decoded = outputs
                    loss = criterion(decoded, images)
                else:
                    loss = criterion(outputs, images)
            else:
                # For classifiers: calculate classification loss and accuracy
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

            # Accumulate loss and count
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

    # Return average loss and accuracy
    if is_autoencoder:
        return running_loss / len(dataloader), correct/total
    else:
        return running_loss / len(dataloader), correct / total


def train(model, model_name, trainloader, optimizer, criterion, device, epochs=10, saved=True, saved_dir="saved", valloader=None, is_autoencoder=False):
    # Main training loop with support for both classifiers and autoencoders
    
    # Setup validation and tracking variables
    use_validation = valloader is not None
    best_val_acc = 0.0  # For classifiers: best accuracy, for autoencoders: best (lowest) loss
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Training loop for specified epochs
    for epoch in range(epochs):
        
        # Training phase
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, is_autoencoder)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase if validation loader provided
        if use_validation:
            val_loss, val_acc = validate(model, valloader, criterion, device, is_autoencoder)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if is_autoencoder:
                # For autoencoders: save model with lowest validation loss
                if epoch == 0 or val_loss < best_val_acc:  # Using best_val_acc variable to store best loss
                    best_val_acc = val_loss
                    if saved:
                        os.makedirs(saved_dir, exist_ok=True)
                        best_model_path = os.path.join(saved_dir, f"{model_name}_best.pt")
                        torch.save(model.state_dict(), best_model_path)
            else:
                # For classifiers: save model with highest validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if saved:
                        os.makedirs(saved_dir, exist_ok=True)
                        best_model_path = os.path.join(saved_dir, f"{model_name}_best.pt")
                        torch.save(model.state_dict(), best_model_path)
    
    # Save final model after training completion
    if saved:
        os.makedirs(saved_dir, exist_ok=True)
        final_model_path = os.path.join(saved_dir, f"{model_name}_final.pt")
        torch.save(model.state_dict(), final_model_path)
    
    return model