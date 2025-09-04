import argparse
import yaml
from types import SimpleNamespace
import torch
import matplotlib.pyplot as plt
import os
import sys 

from models import get_model
from attacks import fgsm, pgd, genetic_attack, few_pixel_attack 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proj1.utils.data_loaders import get_dataloaders, get_dataset_info,denormalize_and_save_image


def load_config(path):
    # Load YAML configuration file
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_cli_config(cfg, args):
    # Override config values with CLI arguments if provided
    # Allows command-line customization without editing config files
    if args.model: cfg["model"]["name"] = args.model
    if args.epsilon: cfg["attack"]["epsilon"] = args.epsilon
    if args.sample_id is not None: cfg["dataset"]["sample_id"] = args.sample_id
    if args.targeted is not None: cfg["attack"]["targeted"] = args.targeted
    return cfg


def run_attack(cfg):
    # Main function to execute adversarial attacks based on configuration
    
    # Setup device and data loading
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _, test_loader, _, test_dataset = get_dataloaders('cifar10')

    # Get dataset information and load model
    info = get_dataset_info(cfg.dataset["name"])
    
    # Initialize model with configuration parameters
    model = get_model(
        name=cfg.model['name'],
        input_shape=(
            cfg.model['params']['channels'], 
            cfg.model['params']['input_size'][0], 
            cfg.model['params']['input_size'][1]
        ),
        device=cfg.device,
        checkpoint_path=cfg.model['checkpoint_path'],
        load_pretrained=True,
        num_filters=cfg.model['params']['num_filters'],
        num_blocks=cfg.model['params']['num_blocks'],
        dropout_rate=cfg.model['params']['dropout_rate'],
        use_batch_norm=cfg.model['params']['use_batch_norm'],
        skip=cfg.model['params']['skip']
    )

    # Set model to evaluation mode
    model.eval()

    # Get specific sample to attack based on configuration
    sample_id = cfg.dataset["sample_id"]
    for x, y in test_loader:
        break

    # Extract and prepare the target sample
    x, y = x[sample_id].unsqueeze(0).to(device), y[sample_id].unsqueeze(0).to(device)
    original_x = x.clone().detach()

    # Get original prediction
    pred = model(x).argmax(dim=1).item()

    # Save original image for comparison
    denormalize_and_save_image(
        original_x,
        path=os.path.join(cfg.output["base_dir"], "original.png"),
        title=f"Original: {pred}"
    )

    # Setup loss function and attack parameters
    criterion = torch.nn.CrossEntropyLoss()
    attack_type = cfg.attack["type"].lower()

    # Execute specific attack based on configuration
    if attack_type == "pgd":
        # Projected Gradient Descent attack
        attacked_label, attacked_image = pgd(
            model, x, y, criterion,
            epsilon=cfg.attack["epsilon"],
            alpha=cfg.attack.get("alpha", 2/255),
            iterations=cfg.attack.get("iterations", 7),
            targeted=cfg.attack.get("targeted", False),
            targetedClass=cfg.attack.get("targeted_class", 6),
            device=device
        )

    elif attack_type == "fgsm":
        # Fast Gradient Sign Method attack
        attacked_label, attacked_image = fgsm(
            model, x, y, criterion,
            epsilon=cfg.attack["epsilon"],
            targeted=cfg.attack.get("targeted", False),
            targetedClass=cfg.attack.get("targeted_class", 0)
        )

    elif attack_type == "one_pixel":
        # Few-pixel attack (sparse perturbation)
        attacked_label, attacked_image = few_pixel_attack(
            model, x, y, criterion,
            num_pixels = cfg.attack.get("num_pixel", 5),
            targeted=cfg.attack.get("targeted", False),
            targetedClass=cfg.attack.get("targeted_class", 0),
            device=device
        )

    elif attack_type == "genetic":
        # Genetic algorithm-based attack
        attacked_image = genetic_attack(
            model, x.cpu(), y.cpu(),
            population_size=cfg.attack.get("population_size", 10),
            generations=cfg.attack.get("generation",50),
            mutation_rate=cfg.attack.get("mutation_rate",0.1),
            device="cpu"
        )
        # Add perturbation to original image
        attacked_image = attacked_image + original_x.cpu()
        # Get prediction for adversarial example
        attacked_label = model(attacked_image.to(device)).argmax(dim=1).item()

    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    # Save adversarial image
    denormalize_and_save_image(
        attacked_image,
        path=os.path.join(cfg.output["base_dir"], f"{attack_type}_attack.png"),
        title=f"Adversarial: {cfg.attack['targeted_class']}"
    )

    # Compute and save difference visualization
    diff = (attacked_image - original_x).squeeze().mean(0).detach().cpu()
    plt.imshow(255 * diff)
    plt.colorbar()
    plt.title("Difference")
    plt.savefig(os.path.join(cfg.output["base_dir"], f"{attack_type}_diff.png"))
    plt.close()


if __name__ == "__main__":
    # Command-line interface setup
    parser = argparse.ArgumentParser(description="Run adversarial attacks on a model.")

    # Define CLI arguments
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--model", type=str, help="Override model name (e.g., CNNclassic, Autoencoder, clip).")
    parser.add_argument("--epsilon", type=float, help="Perturbation strength (e.g., 0.031).")
    parser.add_argument("--sample_id", type=int, help="Index of the sample to attack in the test set.")
    parser.add_argument("--targeted", type=bool, help="Whether the attack is targeted (True/False).")

    # Parse arguments and load configuration
    args = parser.parse_args()
    cfg_dict = load_config(args.config)
    cfg_dict = merge_cli_config(cfg_dict, args)
    cfg = SimpleNamespace(**cfg_dict)

    # Create output directory and run attack
    os.makedirs(cfg.output["base_dir"], exist_ok=True)
    run_attack(cfg)