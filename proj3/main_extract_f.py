import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
from utils.data_utils import load_dataset_hf
from utils.model_utils import load_model, extract_feature
from utils.train_utils import train_svm, train_random_forest, evaluate_model


def extract_features_from_split(model, tokenizer, dataset_split, batch_size=100):
    # Extract features from a dataset split using the pre-trained model
    
    texts = dataset_split['text']
    labels = dataset_split['label']
    
    # Process texts in batches to avoid memory issues
    all_features = []
    for i in range(0, len(texts), batch_size):
        # Extract current batch of texts
        batch_texts = texts[i:i+batch_size]
        # Get feature representations from the model
        batch_features = extract_feature(model, tokenizer, batch_texts)
        all_features.append(batch_features)
    
    # Concatenate all batch features into a single array
    features = np.concatenate(all_features, axis=0)
    labels = np.array(labels)
    
    return features, labels


def save_experiment_results(svm_model, rf_model, scaler, svm_results, rf_results, 
                          best_model, svm_time, rf_time, save_dir="models/baseline"):
    # Save all trained models and experimental results to disk
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save trained SVM model
    with open(os.path.join(save_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
    
    # Save trained Random Forest model
    with open(os.path.join(save_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save feature scaler for consistent preprocessing
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Compile comprehensive results summary
    results_summary = {
        'svm': svm_results,
        'random_forest': rf_results,
        'best_model': best_model,
        'training_times': {'svm': svm_time, 'random_forest': rf_time}
    }
    
    # Save results summary for later analysis
    with open(os.path.join(save_dir, 'results_summary.pkl'), 'wb') as f:
        pickle.dump(results_summary, f)


def print_comparison_table(svm_results, rf_results):
    # Display formatted comparison table of model performances
    
    # Extract accuracy metrics for both models
    svm_train_acc = svm_results['train']['accuracy']
    svm_val_acc = svm_results['validation']['accuracy']
    svm_test_acc = svm_results['test']['accuracy']
    
    rf_train_acc = rf_results['train']['accuracy']
    rf_val_acc = rf_results['validation']['accuracy']
    rf_test_acc = rf_results['test']['accuracy']
    
    # Print formatted comparison table
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Train':<10} {'Validation':<12} {'Test':<10}")
    print("-" * 50)
    print(f"{'SVM':<15} {svm_train_acc:<10.4f} {svm_val_acc:<12.4f} {svm_test_acc:<10.4f}")
    print(f"{'Random Forest':<15} {rf_train_acc:<10.4f} {rf_val_acc:<12.4f} {rf_test_acc:<10.4f}")
    
    # Determine best model based on validation accuracy
    if svm_val_acc > rf_val_acc:
        best_model = "SVM"
        best_val_acc = svm_val_acc
        best_test_acc = svm_test_acc
    else:
        best_model = "Random Forest"
        best_val_acc = rf_val_acc
        best_test_acc = rf_test_acc
    
    # Display best model summary
    print(f"\nBest model: {best_model}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {best_test_acc:.4f}")
    
    return best_model, best_test_acc


def main():
    # Main function to run baseline classification experiments
    
    # Load the Rotten Tomatoes dataset from Hugging Face
    dataset = load_dataset_hf("rotten_tomatoes")
    
    # Load pre-trained model and tokenizer for feature extraction
    model, tokenizer = load_model()
    
    # Extract features for all dataset splits
    X_train, y_train = extract_features_from_split(model, tokenizer, dataset['train'])
    X_val, y_val = extract_features_from_split(model, tokenizer, dataset['validation'])
    X_test, y_test = extract_features_from_split(model, tokenizer, dataset['test'])
    
    # Standardize features for better classifier performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data
    X_val_scaled = scaler.transform(X_val)          # Apply same scaling to validation
    X_test_scaled = scaler.transform(X_test)        # Apply same scaling to test
    
    # Train SVM classifier and measure training time
    start_time = time.time()
    svm_model = train_svm(X_train_scaled, y_train, C=1.0, kernel='rbf')
    svm_time = time.time() - start_time
    
    # Train Random Forest classifier and measure training time
    start_time = time.time()
    rf_model = train_random_forest(X_train_scaled, y_train, n_estimators=100)
    rf_time = time.time() - start_time
    
    # Evaluate both models on all dataset splits
    svm_results = evaluate_model(X_train_scaled, y_train, X_val_scaled, y_val, 
                                X_test_scaled, y_test, svm_model, "SVM")
    
    rf_results = evaluate_model(X_train_scaled, y_train, X_val_scaled, y_val,
                               X_test_scaled, y_test, rf_model, "Random Forest")
    
    # Compare results and determine best model
    best_model, best_test_acc = print_comparison_table(svm_results, rf_results)
    
    # Save all models and results for future use
    save_experiment_results(svm_model, rf_model, scaler, svm_results, rf_results, 
                           best_model, svm_time, rf_time)


if __name__ == "__main__":
    main()