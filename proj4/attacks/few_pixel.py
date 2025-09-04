import torch

def few_pixel_attack(model, x, y, loss_fn, num_pixels=5, epsilon=0.1, targeted=False, targetedClass=None, device='cpu'):
    # Few-pixel attack: modify top-k pixels with highest gradients
    # This attack targets only the most influential pixels based on gradient magnitude
    
    # Clone input tensors and move to specified device
    x_original = x.clone().detach().to(device)
    x_work = x.clone().detach().to(device)
    x_work.requires_grad_(True)  # Enable gradient computation

    # Get original model prediction
    with torch.no_grad():
        original_pred = model(x_original).argmax().item()

    # Set target label based on attack type
    label = torch.tensor([targetedClass]).to(device) if targeted else y

    # Compute gradients with respect to input
    output = model(x_work)
    model.zero_grad()  # Clear any existing gradients
    loss_value = loss_fn(output, label)
    loss_value.backward()  # Backpropagate to get input gradients

    # Extract gradients from input tensor
    gradients = x_work.grad.detach()

    # Find top-k pixels with maximum gradient magnitude
    # Sum gradients across color channels to get per-pixel importance
    abs_grad = gradients.abs().sum(dim=1)  # Shape: (B, H, W)
    batch_size, height, width = abs_grad.shape
    
    # Get indices of top-k most important pixels
    flat_grad = abs_grad.view(batch_size, -1)  # Flatten spatial dimensions
    top_k_values, top_k_indices = torch.topk(flat_grad, num_pixels, dim=1)
    
    # Convert linear indices back to 2D coordinates (height, width)
    top_k_coords = []
    for i in range(num_pixels):
        idx = top_k_indices[0, i]  # Use first batch element
        h = idx // width  # Row coordinate
        w = idx % width   # Column coordinate
        top_k_coords.append((h.item(), w.item()))

    # Create adversarial example by modifying selected pixels
    x_adv = x_original.clone()
    
    # Apply perturbation to each selected pixel across all color channels
    for h, w in top_k_coords:
        for c in range(x_adv.shape[1]):  # Iterate through color channels
            original_val = x_adv[0, c, h, w].item()
            grad_sign = torch.sign(gradients[0, c, h, w])  # Direction of steepest ascent
            
            # Apply perturbation based on attack type
            if targeted:
                # For targeted attacks, move towards target class
                new_val = original_val - epsilon * grad_sign
            else:
                # For untargeted attacks, move away from correct class
                new_val = original_val + epsilon * grad_sign
                
            # Clamp pixel values to valid range [0, 1]
            x_adv[0, c, h, w] = torch.clamp(new_val, 0.0, 1.0)

    # Verify number of pixels actually modified
    diff = torch.abs(x_adv - x_original)
    pixel_mask = diff.sum(dim=1) > 1e-7  # Threshold for detecting changes
    unique_pixels_changed = pixel_mask.sum().item()
    
    # Get final prediction on adversarial example
    with torch.no_grad():
        final_pred = model(x_adv).argmax().item()

    # Return results without verbose output
    return final_pred, x_adv