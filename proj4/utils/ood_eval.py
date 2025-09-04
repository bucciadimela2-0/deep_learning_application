import torch
import torch.nn as nn
from sklearn import metrics

from utils.plot_utils import *
from utils.data_utils import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proj1.utils.data_loaders import get_dataloaders

def compute_scores(model, testloader, device, predict=None, loss=None, score_fun=None):
    # Compute confidence/anomaly scores for OOD detection
    # Different scoring methods for different model types
    
    model.eval()
    scores = []
    
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)

            if score_fun is not None:
                # For classifiers: use confidence-based scoring (max logit)
                output = model(x)
                s = score_fun(output)
                scores.append(s)

            elif loss is not None:
                # For autoencoders: use reconstruction error as anomaly score
                _, prediction = model(x)
                l = loss(x, prediction)
                # Average loss across spatial dimensions, negate for higher=more normal
                s = l.mean(dim=(1, 2, 3))
                scores.append(-s)
            else: 
                raise ValueError("compute_scores richiede loss o score_fun")

    return torch.cat(scores)

def max_logit(logits):
    # Extract maximum logit as confidence score
    # Higher values indicate higher model confidence
    return logits.max(dim=1)[0]

def confusion_matrix(model, testloader, device, num_classes=10): 
    # Compute confusion matrix for classification performance evaluation
    
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            
            # For CNN: output are logits [batch_size, num_classes]
            _, predicted = torch.max(output, 1)
            
            # Update confusion matrix
            for true_label, pred_label in zip(y, predicted):
                cm[true_label, pred_label] += 1
    
    return cm


def evaluate_ood(model, model_type, device, plot_prefix=""):
    # Main OOD evaluation function
    # Compares model behavior on in-distribution vs out-of-distribution data
    
    # Configure loss and prediction functions based on model type
    loss_fn = nn.MSELoss(reduction='none') if model_type == "autoencoder" else nn.CrossEntropyLoss()
    predict_fn = autoencoder_predict_fn if model_type == "autoencoder" else classifier_predict_fn
    score_fun = max_logit if model_type == 'cnn' else None
    
    # Load in-distribution data (CIFAR-10) and out-of-distribution data (fake)
    _, testloader, _, test_dataset = get_dataloaders('cifar10', 'zero', 'zero')
    fakeloader, fakeset = get_fake_data_loader()
    
    # Compute confusion matrix for classification performance
    cm = confusion_matrix(model, testloader, device)
    
    # Compute anomaly scores for both real and fake data
    scores_real = compute_scores(model, testloader, device, predict=predict_fn, loss=loss_fn, score_fun=score_fun)
    scores_fake = compute_scores(model, fakeloader, device, predict=predict_fn, loss=loss_fn, score_fun=score_fun)
    
    # Generate evaluation plots
    confusion_matrix_plot(cm, test_dataset, f'proj4/output_ood/Confusion_matrix_{plot_prefix}.png')
    # histogram_plot(model, testloader, test_dataset, f'proj4/output_ood/histogram_{plot_prefix}.png')  # Optional
    plot_score_distributions(scores_real, scores_fake, f'proj4/output_ood/scores_{plot_prefix}.png')
    plot_roc_curve(scores_real, scores_fake, f'proj4/output_ood/ROC_curve_{plot_prefix}.png')

def classifier_predict_fn(output):
    # Prediction function for classification models
    # Returns predicted class indices
    return output.argmax(1)

def autoencoder_predict_fn(output):
    # Prediction function for autoencoder models
    # Returns reconstructed images (decoder output)
    _, x_rec = output
    return x_rec