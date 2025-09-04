A deep learning project implementing CNN and MLP models, with tools for Grad-CAM visualization, training management, and evaluation under degradation or adversarial conditions.
'''
📂 Project Structure
proj1/
│── config/               # Configuration files (YAML)
│   ├── cnn/              # CNN-specific configs
│   ├── mlp/              # MLP-specific configs
│   └── gradcam.yaml      # Grad-CAM configuration
│
│── gradcam_results/      # Grad-CAM analysis results
│── gradcam_results_attack/ # Grad-CAM results under attack
│
│── models/               # Model definitions
│   ├── __init__.py
│   ├── cnn.py
│   └── mlp.py
│
│── utils/                # Utility functions
│   ├── __init__.py
│   ├── config_utils.py
│   ├── data_loaders.py
│   ├── early_stopping.py
│   ├── gradcam_analyzer.py
│   ├── gradcam_utils.py
│   └── train.py
│
│── degradation_main.py   # Main script for degradation experiments
│── grad_cam_main.py      # Main script for Grad-CAM experiments
│── requirements.txt      # Project dependencies
│── resume.txt            # Notes or previous experiment logs

'''
