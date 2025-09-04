import argparse
import yaml
from types import SimpleNamespace


def load_config(path):
    # Load configuration from YAML file
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def create_parser():
    # Create argument parser with all configuration options
    parser = argparse.ArgumentParser(description="Train neural network with configurable options")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    # Regularization options
    parser.add_argument("--augment", type=bool, default=None, 
                       help="Enable data augmentation (overrides config)")
    parser.add_argument("--batch-norm", type=bool, default=None,
                       help="Enable batch normalization (overrides config)")
    parser.add_argument("--dropout", type=float, default=None,
                       help="Dropout rate (overrides config)")
    
    # Training options
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (overrides config)")
    
    # Model options
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (overrides config)")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name (overrides config)")
    
    # Other options
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--save-model", type=bool, default=None,
                       help="Save model after training (overrides config)")
    
    return parser


def update_config_from_args(config_dict, args):
    # Update config dictionary with command line arguments
    
    # Dataset configuration
    if args.augment is not None:
        config_dict["dataset"]["augment"] = args.augment
    if args.dataset is not None:
        config_dict["dataset"]["name"] = args.dataset
    
    # Model configuration
    if args.model is not None:
        config_dict["model"]["name"] = args.model
    if args.batch_norm is not None:
        # Initialize params dict if it doesn't exist
        if "params" not in config_dict["model"]:
            config_dict["model"]["params"] = {}
        config_dict["model"]["params"]["use_batch_norm"] = args.batch_norm
    if args.dropout is not None:
        # Initialize params dict if it doesn't exist
        if "params" not in config_dict["model"]:
            config_dict["model"]["params"] = {}
        config_dict["model"]["params"]["dropout_rate"] = args.dropout
    
    # Training configuration
    if args.lr is not None:
        config_dict["lr"] = args.lr
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.seed is not None:
        config_dict["seed"] = args.seed
    
    # Other configuration
    if args.no_wandb:
        config_dict["wandb"]["enabled"] = False
    if args.save_model is not None:
        config_dict["save_model"] = args.save_model
    
    return config_dict


def parse_config():
    # Parse command line arguments and load configuration
    parser = create_parser()
    args = parser.parse_args()
    
    # Load base configuration from YAML file
    config_dict = load_config(args.config)
    # Update configuration with command line overrides
    config_dict = update_config_from_args(config_dict, args)
    # Convert dictionary to namespace for easier access
    config = SimpleNamespace(**config_dict)
    
    return config, args


def print_config_summary(config):
    # Print a summary of the current configuration
    
    # Header
    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    
    # Basic configuration
    print(f"Dataset: {config.dataset['name']}")
    print(f"Model: {config.model['name']}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"Seed: {config.seed}")
    print(f"Data Augmentation: {config.dataset['augment']}")
    
    # Model specific parameters
    if "params" in config.model:
        params = config.model["params"]
        if "use_batch_norm" in params:
            print(f"Batch Normalization: {params['use_batch_norm']}")
        if "dropout_rate" in params:
            print(f"Dropout Rate: {params['dropout_rate']}")
    
    # Additional configuration
    print(f"W&B Logging: {config.wandb['enabled']}")
    print(f"Save Model: {config.save_model}")
    print("=" * 50)


def get_model_name(config):
    # Get model name for saving purposes
    # Use custom name from params if available, otherwise use default model name
    model_name = config.model["params"].get("name", config.model["name"])
    return f"{model_name}_{config.dataset['name']}"


def validate_config(config):
    # Validate configuration parameters
    
    # Check for required fields
    required_fields = ["seed", "epochs", "batch_size", "lr", "dataset", "model"]
    
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Validate dataset configuration
    if "name" not in config.dataset:
        raise ValueError("Dataset name is required")
    
    # Validate model configuration
    if "name" not in config.model:
        raise ValueError("Model name is required")
    
    # Validate numeric parameters
    if config.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if config.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if config.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    # Validate dropout rate if present
    if "params" in config.model and "dropout_rate" in config.model["params"]:
        dropout_rate = config.model["params"]["dropout_rate"]
        if not (0 <= dropout_rate < 1):
            raise ValueError("Dropout rate must be between 0 and 1")
    
    return True