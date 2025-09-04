# deep_learning_application

Laboratory repository for the Deep Learning Applications course, featuring hands-on experiments across Computer Vision, Natural Language Processing, and Adversarial Machine Learning domains.


## :test_tube: Lab1 - Convolutional Neural Networks
The first laboratory studies MLP degradation patterns and vanishing gradients across different activation functions and regularization techniques. It then compares standard CNNs against versions with skip connections, demonstrating how residual connections solve the degradation problem. Finally, Grad-CAM analysis is applied to the best model on both clean and adversarially perturbed images to reveal attention pattern changes under attack conditions.
> **Experimental Results**: All experiments and training metrics are tracked and visualized at: [wandb](https://wandb.ai/martina-buccioni98-unifi/deep-learning-application?nw=nwusermartinabuccioni98)

<details>
<summary>Let's break the ice with MLP </summary>
Among all the experiments conducted to study MLPs, two caught my attention. The first one focuses on the vanishing gradient problem in MLPs (to be fair, without any type of regularization). The second one, instead, focuses on normalizations.
<div align="center">
<img src="plots/mlp_activation_function.png" alt="Training Loss by Activation Function" width="250"/>
<img src="plots/normalizzazione.png" alt="Validation Loss by Regularization Method" width="250"/>
<p><em>Left: Training loss comparison across activation functions | Right: Validation loss for different regularization strategies</em></p>
</div>
As shown in the left plot, the combination of saturating activation functions (sigmoid, tanh) and deep network architectures creates a multiplicative effect, causing gradients to diminish exponentially with each layer. This explains why early layers struggle to receive meaningful updates, resulting in slower or stalled training.

The right plot illustrates the impact of different regularization strategies. These results highlight that data augmentation can act as a powerful form of regularization, often outperforming architectural modifications in improving validation performance.
> **Dataset Augmentation in Feature Space**
> Terrance DeVries, Graham W. Taylor, ICLR 2017
</details>

<details>
<summary>To skip or not to skip - CNN  </summary>
<div align="center">
<img src="plots/skipornottoskip_loss.png" width="250"/>
<img src="plots/skipotnottoskip.png"  width="250"/>
<p><em>Learning curves</em></p>
</div>

| Architecture | Size | Depth | Final Accuracy | 
|--------------|------|-------|----------------|
| CNN | Small | [2,2] | 68% |
| CNN | Medium | [5,5] | 77% | 
| CNN | Large | [7,7] | 75% | 
| CNN + skip | Small | [2,2] | 63% | 
| CNN + skip | Medium | [5,5] | 79% | 
| CNN + skip | Large | [7,7] | **82%** | 

These results confirm the fundamental insight from ResNet - that skip connections solve the degradation problem by allowing gradients to flow directly through identity mappings, enabling effective training of very deep networks.
</details>
<details>
<summary>Gradcam: what does my best model sees? </summary>
Grad-CAM (Gradient-weighted Class Activation Mapping) analysis was applied to the best-performing CNN to understand what regions the model focuses on for classification decisions. We want to show how the attention focuses over adversarial examples.

<div align="center">
<img src="proj1/gradcam_results/sample_4_gradcam.png" alt="Grad-CAM Airplane Analysis" width="250"/>
<img src="proj1/gradcam_results_attack/sample_4_gradcam.png" width="250"/>
</div>
The analysis reveals how adversarial perturbations dramatically alter the model's attention patterns. In the original images, the model focuses on semantically relevant features; however, under adversarial attacks, attention either scatters to irrelevant regions or concentrates on attack-induced artifacts.
This phenomenon is clearly illustrated in other two cool examples, namely image 1 and image 7.
</details>

## :test_tube: Lab3 - Transformers and NLP
<details>




</details>


## :test_tube: Lab4 - Adversarial Learning and Out Of Distribution
This laboratory explores model security and robustness through three sequential investigations. First, we implement various adversarial attack methods - FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent), one-pixel attacks, and genetic algorithm-based perturbations.
Second, we investigate out-of-distribution (OOD) detection capabilities by comparing two approaches: reconstruction-based detection using autoencoders versus confidence-based detection with CNNs. Finally, we enhance model robustness through adversarial training, incorporating adversarial examples generated from our attack methods into the training process. This demonstrates how exposure to adversarial samples during training can significantly improve model resilience against future attacks.

<details>
<summary>Let's hack it : adversarial attacks</summary>
- FGSM : 
<div align="center">
<img src="proj4/output_adv/fgsm/fgsm_attack.png" alt="FGSM Attack" width="300"/>
<img src="proj4/output_adv/fgsm/fgsm_diff.png" alt="One-Pixel Attack" width="400"/>
<p><em>Left: FGSM distributed perturbations | Right: FGSM difference patterns</em></p>
</div>

- PGD : 
<div align="center">
<img src="proj4/output_adv/pgd/pgd_attack.png" alt="FGSM Attack" width="300"/>
<img src="proj4/output_adv/pgd/pgd_diff.png" alt="One-Pixel Attack" width="400"/>
<p><em>Left: PGD distributed perturbations | Right: PGD difference patterns</em></p>
</div>
- Few-Pixel attack  : 
<div align="center">
<img src="proj4/output_adv/one_pixel/one_pixel_attack.png" alt="FGSM Attack" width="300"/>
<img src="proj4/output_adv/one_pixel/one_pixel_diff.png" alt="One-Pixel Attack" width="400"/>
<p><em>Left: Few_pixel distributed perturbations | Right: Few-pixwl difference patterns</em></p>
</div>
- Genetic attack  : 
<div align="center">
<img src="proj4/output_adv/genetic/genetic_attack.png" alt="FGSM Attack" width="300"/>
<img src="proj4/output_adv/genetic/genetic_diff.png" alt="One-Pixel Attack" width="400"/>
<p><em>Left: Genetic distributed perturbations | Right: Genetic difference patterns</em></p>
</div>


</details>
