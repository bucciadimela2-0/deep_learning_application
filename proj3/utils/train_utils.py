import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List, Tuple, Any
import time


def train_svm(x_train, y_train, C=1.0, kernel='rbf'):
    # Train Support Vector Machine classifier
    
    # Initialize SVM with specified hyperparameters
    svm_model = SVC(C=C, kernel=kernel, random_state=42)
    # Fit model to training data
    svm_model.fit(x_train, y_train)
    
    return svm_model


def train_random_forest(x_train, y_train, n_estimators=100, max_depth=None):
    # Train Random Forest classifier
    
    # Initialize Random Forest with specified parameters
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,  # Number of trees in the forest
        max_depth=max_depth,        # Maximum depth of trees (None = unlimited)
        random_state=42,            # For reproducible results
        n_jobs=-1                   # Use all available CPU cores
    )
    # Fit model to training data
    rf_model.fit(x_train, y_train)
    
    return rf_model


def evaluate_model(x_train, y_train, x_val, y_val, x_test, y_test, model, model_name: str):
    # Evaluate a trained model on train, validation, and test splits
    
    results = {}
    
    # Evaluate model performance on all data splits
    for split_name, x_split, y_split in [
        ('train', x_train, y_train),
        ('validation', x_val, y_val),
        ('test', x_test, y_test)
    ]:
        # Get predictions for current split
        y_pred = model.predict(x_split)
        # Calculate accuracy score
        accuracy = accuracy_score(y_split, y_pred)
        
        # Store results for this split
        results[split_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_split
        }
    
    return results