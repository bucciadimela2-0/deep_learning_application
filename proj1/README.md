A deep learning project implementing CNN and MLP models, with tools for Grad-CAM visualization, training management, and evaluation under degradation or adversarial conditions.
<details>
<summary>🎯 Objectives<\summary>
The exercises are designed to explore the behavior of different neural network architectures and techniques, focusing on training dynamics, regularization, and model interpretability.

- *MLP (Multi-Layer Perceptron)**
  - Analyze the **vanishing gradient problem**
  - Compare **deep vs. medium vs. shallow** networks
  - Evaluate the effect of different **regularization strategies**:
    - Dropout (low, medium, aggressive)
    - Batch Normalization
    - Data Augmentation

- *CNN (Convolutional Neural Networks)**
  - Test **data augmentation as regularization**
  - Show that **deeper does not always mean better**
  - Demonstrate improvements with **skip connections**

- *Grad-CAM (Model Interpretability)**
  - Generate **standard Grad-CAM visualizations**
  - Assess robustness under **FGSM adversarial attacks**
    - Notable examples on digits **4, 1, and 7**
<\details>
> **Note:** All experiments are explained in detail in the main `README.md` located in the root of the repository.
<details>
<summary>📂 Project Structure<\summary>
  
```
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
```

<\details>
