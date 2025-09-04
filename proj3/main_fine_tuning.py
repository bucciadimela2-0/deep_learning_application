import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from utils.data_utils import load_dataset_hf, tokenize_dataset
from utils.model_utils import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def run_fine_tuning(
    dataset_name="rotten_tomatoes",
    model_name="distilbert-base-uncased",
    output_dir="models/distilbert_finetuned",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=512,
    device="cuda"  # "cuda" o "cpu"
):
    # Main function to fine-tune a pre-trained transformer model for classification

    # Define metrics computation function for evaluation
    def compute_metrics(eval_pred):
        # Extract predictions and true labels
        predictions, labels = eval_pred
        # Convert logits to predicted class indices
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    # Load dataset from Hugging Face
    dataset = load_dataset_hf(dataset_name)
    # Load pre-trained model and tokenizer with classification head
    model, tokenizer = load_model(model_name=model_name, num_labels=2, task="classification")

    # Define safe tokenization function with proper padding and truncation
    def safe_tokenize(batch):
        # Tokenize text with consistent max length and padding
        return tokenizer(
            batch['text'],
            padding='max_length',    # Pad all sequences to max_length
            truncation=True,         # Truncate sequences longer than max_length
            max_length=max_length
        )

    # Apply tokenization to entire dataset in batches
    tokenized_dataset = dataset.map(safe_tokenize, batched=True)
    # Set format to PyTorch tensors for required columns
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,                          
        num_train_epochs=num_epochs,                    
        per_device_train_batch_size=batch_size,         
        per_device_eval_batch_size=batch_size,          
        learning_rate=learning_rate,                    
        weight_decay=0.01,                             
        save_strategy="epoch",                         
        eval_strategy="epoch",                          
        load_best_model_at_end=True,                  
        metric_for_best_model="accuracy",               
        greater_is_better=True,                         
        report_to=[],                                  
        seed=42                                        
    )

    # Initialize Trainer with model, datasets, and training configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    # Start fine-tuning process
    trainer.train()

    # Evaluate model performance on validation set
    eval_result = trainer.evaluate()
    # Evaluate model performance on test set
    test_result = trainer.evaluate(eval_dataset=tokenized_dataset["test"])

    # Save fine-tuned model and tokenizer to specified directory
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Return evaluation results for both validation and test sets
    return {"validation": eval_result, "test": test_result}


def main():
    # Main execution function to run fine-tuning with specified parameters
    
    results = run_fine_tuning(
        dataset_name="rotten_tomatoes",          
        model_name="distilbert-base-uncased",    
        output_dir="models/distilbert_finetuned", 
        num_epochs=3,                            
        batch_size=8,                           
        learning_rate=2e-5,                     
        max_length=512,                         
        device="cuda"                            
    )
    
    # Display final results
    print("Validation results:", results["validation"])
    print("Test results:", results["test"])


if __name__ == "__main__":
    # Configure GPU visibility (use only GPU 0)
    #import os
   # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Run the main fine-tuning pipeline
    main()