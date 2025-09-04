import torch
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def load_model(model_name: str = "distilbert-base-uncased", num_labels: int = 2, task = None, cache_dir= "../"):
    # Load pre-trained model and tokenizer based on task type
    
    # Load tokenizer with caching
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = os.path.join(cache_dir, "tokenizers"))
    
    # Load appropriate model based on task
    if task == "classification":
        # Load classification model with specified number of output labels
        model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=os.path.join(cache_dir, "models"))
    elif task == "seq2seq":
        # Load sequence-to-sequence model for generation tasks
         model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=os.path.join(cache_dir, "models")
        )
    else:
        # Load base model for feature extraction or custom tasks
        model = AutoModel.from_pretrained(model_name,
        cache_dir=os.path.join(cache_dir, "models"))
    
    return model, tokenizer


def extract_feature(model, tokenizer, texts, max_length=128, batch_size=32):
    # Extract features from texts using the model's hidden representations
    
    # Determine device: use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    
    # Create feature extraction pipeline
    feature_extractor = pipeline(
    model=model, 
    tokenizer=tokenizer, 
    task="feature-extraction",
    framework="pt", 
    device=device, 
    batch_size=batch_size,
    tokenize_kwargs=dict(max_length=max_length, truncation=True))

    # Convert single string to list for consistent processing
    if isinstance(texts, str):
        texts = [texts]

    # Extract features from all texts
    extracts = feature_extractor(texts, return_tensors='pt')
    features = []
    
    # Process each text's features
    for e in extracts:
        # Extract [CLS] token representation (first token in sequence)
        # Shape: [seq_len, hidden_size] -> extract [hidden_size]
        cls_feature = e[0].numpy()[0]  # First token is [CLS] token
        features.append(cls_feature)
    
    # Stack all features into a single numpy array
    return np.vstack((features))