"""
Main script for Grad-CAM and adversarial analysis.
"""
import torch
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config_utils import parse_config, validate_config
from utils import get_dataset_info, set_seed
from models import get_model
from utils.gradcam_analyzer import ModelAnalyzer
from utils.gradcam_utils import (
    get_test_dataset,
    load_checkpoint,
    log_analysis_config,
    log_model_info
)


def run_analysis(analyzer, config, test_loader, logger):
    # Run the main analysis loop for Grad-CAM and adversarial attacks
    
    # Get number of samples to analyze from config
    num_samples = getattr(config, 'num_samples', 5)
    
    # Create iterator once and advance through it to avoid resetting
    test_iter = iter(test_loader)
    
    # Process each sample
    for idx in range(num_samples):
        
        try:
            # Get next sample from iterator
            image, true_label = next(test_iter)
        except StopIteration:
            # If we run out of samples, create new iterator
            test_iter = iter(test_loader)
            image, true_label = next(test_iter)
            
        # Move image to device and keep original copy
        image = image[0].to(analyzer.device)
        original_image = image.clone()
        
        # Get initial model prediction
        pred_class, confidence = analyzer.get_prediction(image)
        # Determine dataset type for logging
        dataset_type = "MNIST digit" if getattr(config, 'ood', False) else analyzer.cifar_classes[true_label.item()]
        
        # Apply FGSM adversarial attack if enabled
        if getattr(config, 'fgsm', False):
            # Get attack parameters from config
            target_class = getattr(config, 'target_class', 2)
            epsilon = getattr(config, 'epsilon', 0.03)
            
            # Generate adversarial example
            image = analyzer.fgsm_attack(image, target_class, epsilon)
            
            # Visualize attack perturbation if requested
            if getattr(config, 'show_perturbation', False):
                save_dir = getattr(config, 'save_dir', 'results')
                attack_path = f'{save_dir}/sample_{idx + 1}_attack.png'
                analyzer.visualize_attack(original_image, image, epsilon, attack_path)
            
            # Get prediction after attack
            pred_class, confidence = analyzer.get_prediction(image)
        
        # Apply Grad-CAM analysis
        try:
            # Generate class activation map
            cam, gradcam_class = analyzer.apply_gradcam(image)
            title = f"Grad-CAM: {analyzer.cifar_classes[gradcam_class]} (conf: {confidence:.3f})"
            
            # Save Grad-CAM visualization
            save_dir = getattr(config, 'save_dir', 'results')
            save_path = f'{save_dir}/sample_{idx + 1}_gradcam.png'
            alpha = getattr(config, 'alpha', 0.6)  # Overlay transparency
            
            analyzer.visualize_gradcam(image, cam, title, save_path, alpha)
            
        except Exception as e:
            # Continue with next sample if Grad-CAM fails
            continue
        
        

def main():
    # Main function for Grad-CAM and adversarial analysis
    
    # Parse and validate configuration from command line and config file
    config, args = parse_config()
    validate_config(config)
    
    # Setup logging for this analysis session
    logger = log_analysis_config(config)
    
    # Setup device and ensure reproducible results
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset information using existing utilities
    dataset_name = config.dataset.get("name", "cifar10")
    info = get_dataset_info(dataset_name)

    # Initialize model with dataset-specific parameters
    model = get_model(
        config.model["name"],
        input_shape=(info["channels"], info["height"], info["width"]),
        num_classes=info["classes"], 
        **config.model.get("params", {})  # Pass additional model parameters
    ).to(device)

    # Log model information
    log_model_info(model, config, logger)

    # Load trained model checkpoint if provided
    if hasattr(config, 'checkpoint') and config.checkpoint:
        if not load_checkpoint(model, config.checkpoint, device, logger):
            return  # Exit if checkpoint loading fails

    # Initialize analyzer with model and device
    analyzer = ModelAnalyzer(model, device, logger)

    # Get test dataset using existing utilities (supports OOD testing)
    test_loader = get_test_dataset(
        dataset_name,
        getattr(config, 'ood', False),  # Enable OOD testing if specified
        logger
    )

    # Run the main analysis loop
    try:
        run_analysis(analyzer, config, test_loader, logger)
        
    except KeyboardInterrupt:
        # Handle user interruption 
        pass
    except Exception as e:
        # Handle unexpected errors
        raise


if __name__ == "__main__":
    main()