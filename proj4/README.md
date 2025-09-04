
A deep learning project implementing adversarial attack methods, out-of-distribution detection, and model robustness evaluation with comprehensive visualization tools.

<details>
<summary>🎯 Objectives</summary>

The project focuses on understanding model vulnerabilities and robustness through adversarial attacks and out-of-distribution detection:

- **Adversarial Attacks**
  - Implement **FGSM (Fast Gradient Sign Method)** attacks
  - Develop **PGD (Projected Gradient Descent)** attacks  
  - Create **few-pixel attacks** for sparse perturbations
  - Apply **genetic algorithm-based** adversarial generation

- **Out-of-Distribution Detection**
  - Evaluate model behavior on **unseen data distributions**
  - Compare **CNN vs Autoencoder** approaches for anomaly detection
  - Generate **ROC curves** and performance metrics

- **Model Analysis & Visualization**
  - Create comprehensive **attack visualizations**
  - Generate **confusion matrices** and performance plots
  - Analyze **score distributions** for normal vs anomalous data

</details>

> **Note:** This project explores the intersection of adversarial machine learning and anomaly detection, providing tools for comprehensive model robustness evaluation.

<details>
<summary>📂 Project Structure</summary>

```
proj4/
├── attacks/                         # Adversarial attack implementations
│   ├── __init__.py                  # Attack method exports
│   ├── few_pixel.py                 # Sparse pixel-based attacks
│   ├── fgsm.py                      # Fast Gradient Sign Method
│   ├── genetic_attack.py            # Genetic algorithm attacks
│   └── pgd.py                       # Projected Gradient Descent
│
├── config/                          # Configuration files
│   ├── adv_attack/                  # Adversarial attack configs
│   └── ood/                         # OOD detection configs
│
├── models/                          # Model architectures
│   ├── __init__.py                  # Model factory
│   ├── autoencoder.py               # Autoencoder for anomaly detection
│   └── cnn.py                       # CNN classifier
│
├── output_adv/                      # Adversarial attack results
├── output_ood/                      # OOD detection results
│
├── utils/                           # Utility functions
│   ├── data_utils.py                # Data loading and preprocessing
│   ├── ood_eval.py                  # OOD evaluation metrics
│   └── plot_utils.py                # Visualization tools
│
├── main_adv.py                      # Adversarial attack orchestration
└── main_ood.py                      # OOD detection experiments
 
```

</details>


<details>
<summary> Core Components – Adversarial Attacks</summary>

This project includes implementations of several adversarial attack methods for neural networks.

FGSM (Fast Gradient Sign Method)
- Single-step attack using the gradient sign:  
  `x_adv = x + ε * sign(∇_x J(θ,x,y))`
  > *[Explaining and Harnessing Adversarial Examples]*  
  Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy, ICLR 2015

 PGD (Projected Gradient Descent)
- Multi-step FGSM with projection into an ε-ball for stronger attacks
 > *[Towards Deep Learning Models Resistant to Adversarial Attacks]* 
 >Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu, ICLR 2018

#Few-Pixel Attack
- Sparse perturbations targeting only high-gradient pixels
 >*[One Pixel Attack for Fooling Deep Neural Networks]*  
 > Jiawei Su, Danilo Vasconcellos Vargas, Kouichi Sakurai, IEEE TEC 2019

 Genetic Algorithm Attack
- Evolutionary optimization of perturbations through selection and mutation
 >*[Reference: *Generating Natural Language Adversarial Examples]*
 > Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang, EMNLP 2018

</details>
