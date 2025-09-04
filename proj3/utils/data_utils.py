import os
from datasets import load_dataset
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Any
import numpy as np

def load_dataset_hf(dataset_name: str, subset = None , cache_dir: Optional[str] = None, data_dir = "../data"):
    # Load HuggingFace dataset with caching
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Set default cache directory based on dataset name
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, dataset_name.replace("/", "_"))
    
    # Load dataset with or without subset
    if subset:
        dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
    else: 
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset


class CustomDataset(Dataset):
    # Custom dataset for handling different NLP tasks
    
    def __init__(self, dataset, tokenizer, task_type="sentiment", max_length=200, source_lang="de", target_lang="en"):
        # Initialize dataset with task-specific parameters
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def __len__(self):
        # Return dataset length
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get and preprocess item at given index
        item = self.dataset[idx]
        return pre_processing(
            item, 
            self.tokenizer, 
            task_type=self.task_type,
            max_length=self.max_length,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )


def data_loader(
        dataset,
        tokenizer,
        batch_size: int = 16,
        shuffle: bool = True,
        task_type: str = "sentiment",
        max_length: int = 512,
        source_lang: str = "en",
        target_lang: str = "de",
        num_workers: int = 0
    ):
    # Create DataLoader for given dataset and task
    
    # Create custom dataset wrapper
    custom_dataset = CustomDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        task_type=task_type,
        max_length=max_length,
        source_lang=source_lang,
        target_lang=target_lang
    )
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        custom_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


def collate_fn(batch):
    # Custom collate function to handle batching of different data types
    collated = {}
    
    # Process each key in the batch items
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            # Stack tensors along batch dimension
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            # Keep non-tensors (like strings) as lists
            collated[key] = [item[key] for item in batch]
    
    return collated


def pre_processing(item, tokenizer, task_type = "sentiment", padding = 'max_length', max_length = 200, source_lang = "de", target_lang = "en",  return_attention_mask=False):
    # Preprocess single item based on task type
    
    if task_type == "sentiment":
        # Process sentiment classification data
        text = item['text']
        label = item['label']
        
        # Tokenize text with specified parameters
        encoding = tokenizer(
            text,
            truncation = True,
            padding = padding,
            max_length = max_length, 
            return_tensors ='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text  # Keep original text for debugging
        }
        
    elif task_type == "translation":
        # Process translation data
        translation_dict = item['translation']
        source_text = translation_dict[source_lang]
        target_text = translation_dict[target_lang]
        
        # Tokenize source text
        source_encoding = tokenizer(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # Tokenize target text for labels
        target_encoding = tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(0),
            'source_text': source_text,
            'target_text': target_text
        }


def tokenize_dataset(dataset, tokenizer, max_length=512):
    # Tokenize entire dataset using preprocessing function
    
    def tokenize_fn(examples):
        # Apply preprocessing with sentiment task settings
        result = pre_processing(examples, tokenizer, max_length = max_length, task_type="sentiment", padding = False)
        return result

    # Apply tokenization to all examples
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=False
    )
    return tokenized_dataset


def dataset_info(dataset):
    # Display comprehensive dataset information
    
    # Calculate total examples across all splits
    total_examples = sum(len(dataset[split]) for split in dataset.keys())
    
    # Show split information
    splits_info = [f"{split}: {len(dataset[split]):,}" for split in dataset.keys()]
    
    # Display feature information
    first_split = list(dataset.keys())[0]
    features = list(dataset[first_split].features.keys())
    
    # Show example data structure
    example = dataset[first_split][2]
    for key, value in example.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like translation pairs)
            for lang, text in value.items():
                if len(text) > 100:
                    pass  # Truncate long text for display
                else:
                    pass
        elif isinstance(value, str) and len(value) > 300:
            pass  # Truncate long strings
        else:
            pass


def test_dataloader():
    # Test function to verify dataloader functionality
    
    # Initialize tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test sentiment classification dataloader
    dataset_rotten = load_dataset_hf("rotten_tomatoes")
    
    sentiment_loader = data_loader(
        dataset=dataset_rotten['train'],
        tokenizer=tokenizer,
        batch_size=4,
        task_type="sentiment",
        max_length=128
    )
    
    # Get and examine sample batch
    sample_batch = next(iter(sentiment_loader))
    
    # Test translation dataloader
    dataset_opus = load_dataset_hf("Helsinki-NLP/opus-100", subset='de-en')
    
    translation_loader = data_loader(
        dataset=dataset_opus['train'],
        tokenizer=tokenizer,
        batch_size=2,
        task_type="translation",
        max_length=64,
        source_lang="de",
        target_lang="en"
    )
    
    # Get and examine translation sample
    sample_batch = next(iter(translation_loader)