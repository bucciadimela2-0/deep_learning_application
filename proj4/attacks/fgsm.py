import torch

def fgsm(model, attacked_image_x, attacked_label_y, loss, targeted=False, targetedClass=None, epsilon=2/255, budget=8/255, device='cpu'):
    # Fast Gradient Sign Method (FGSM) attack implementation
    # Iteratively applies small perturbations in the direction of the gradient sign
    
    # Clone and prepare input image for gradient computation
    attacked_image_x = attacked_image_x.clone().detach().to(device)
    attacked_image_x.requires_grad = True  
    output = model(attacked_image_x)

    # Early termination checks for edge cases
    if not targeted and output.argmax().item() != attacked_label_y.item():
        # For untargeted attacks, if model is already wrong, no attack needed
        return output.argmax(), attacked_image_x

    if targeted and attacked_label_y.item() == targetedClass:
        # For targeted attacks, if ground truth equals target, attack is meaningless
        return output.argmax(), attacked_image_x

    # Initialize attack loop variables
    done = False
    n = 0

    # Iterative FGSM attack loop
    while not done:
        # Enable gradient computation for current iteration
        attacked_image_x.requires_grad = True
        output = model(attacked_image_x)
        model.zero_grad()

        # Set loss target based on attack type
        label = targetedClass if targeted else attacked_label_y
        l = loss(output, torch.tensor([label]).to(output.device))
        l.backward()  # Compute gradients

        # Create perturbation using gradient sign
        perturbation = epsilon * torch.sign(attacked_image_x.grad)
        
        # Apply perturbation in appropriate direction
        if targeted:
            # For targeted attacks: move towards target class (negative gradient)
            attacked_image_x = attacked_image_x - perturbation
        else:
            # For untargeted attacks: move away from correct class (positive gradient)
            attacked_image_x = attacked_image_x + perturbation

        # Update iteration counter and detach gradients
        n += 1
        attacked_image_x = attacked_image_x.detach()

        # Get current prediction
        pred = output.argmax().item()

        # Check success conditions for untargeted attack
        if not targeted and pred != attacked_label_y.item():
            done = True

        # Check success conditions for targeted attack
        if targeted and pred == targetedClass:
            done = True

        # Check if budget is exhausted
        if n * epsilon >= budget:
            break

    return output.argmax(), attacked_image_x