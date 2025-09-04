import sys
import os

# Get the project root directory (dla24_marbuc)
current_dir = os.path.dirname(os.path.abspath(__file__))  # proj1/utils
proj1_dir = os.path.dirname(current_dir)  # proj1
project_root = os.path.dirname(proj1_dir)  # dla24_marbuc

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Added to path: {project_root}")  # Debug line

# Import from utils_global
from utils_global.model_utils import save_checkpoint, load_model, set_seed
from utils_global.wandb_utils import init_wandb, log_metrics, log_images, watch_model

# Import local modules
from .early_stopping import EarlyStopping
from .data_loaders import get_dataloaders, get_dataset_info

from .gradcam_analyzer import ModelAnalyzer
from .gradcam_utils import (
  
    get_test_dataset,
    load_checkpoint,
    log_analysis_config,
    
    log_model_info
)

__all__ = [
    'ModelAnalyzer',
    'log_analysis_config', 
    'get_test_dataset',
    'load_checkpoint',
    'check_gradcam_support',
    
    'log_model_info'
]