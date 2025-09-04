import torch

def pgd(model, x, y, loss, epsilon=8/255, alpha=2/255, iterations=7, targeted=False, targetedClass=None, device='cpu'):
    # Projected Gradient Descent (PGD) attack implementation
    # Multi-step attack that iteratively applies small perturbations while staying within epsilon ball
    
    # Clone and prepare input tensors
    x = x.clone().detach().to(device)
    attacked_image_x = x.clone().detach()

    # Set target label based on attack type
    label = targetedClass if targeted else y

    # Iterative attack loop
    for i in range(iterations):
        # Enable gradient computation for current iteration
        attacked_image_x.requires_grad_(True)

        # Forward pass and gradient computation
        output = model(attacked_image_x)
        model.zero_grad()
        loss_value = loss(output, torch.tensor([label]))
        loss_value.backward()

        # Check for gradient computation issues
        if attacked_image_x.grad is None:
            raise ValueError("Gradient is None. Check if requires_grad is set on the correct tensor.")

        # Create perturbation using gradient sign (FGSM step)
        perturbation = alpha * torch.sign(attacked_image_x.grad)

        # Apply perturbation based on attack type
        if targeted:
            # For targeted attacks: move towards target class
            attacked_image_x = attacked_image_x - perturbation
        else:
            # For untargeted attacks: move away from correct class
            attacked_image_x = attacked_image_x + perturbation

        # Project back into epsilon ball around original image
        # This ensures perturbation magnitude doesn't exceed epsilon
        attacked_image_x = torch.max(torch.min(attacked_image_x, x + epsilon), x - epsilon)
        
        # Detach to prevent gradient accumulation across iterations
        attacked_image_x = attacked_image_x.detach()

    # Get final prediction on adversarial example
    pred = model(attacked_image_x).argmax()
    return pred, attacked_image_x