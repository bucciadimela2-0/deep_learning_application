from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import torch
from evaluate import load
import logging
import json
import os
from datetime import datetime

# Force single GPU usage - No DataParallel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger(__name__)

class GermanCorrector:
    def __init__(self, model_name="t5-small", log_file="corrections.jsonl"):
        # German grammar corrector with single GPU - No DataParallel
        
        # Force single device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.gec_pipeline = None
        self.log_file = log_file

        # Evaluation metrics (optional)
        self.bleu = load("bleu")
        self.rouge = load("rouge")

        # Initialize log file
        self._initialize_log_file()

    def _initialize_log_file(self):
        # Initialize correction log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                header = {
                    "timestamp": datetime.now().isoformat(),
                    "session_start": True,
                    "model": self.model_name,
                    "device": str(self.device),
                }
                f.write(json.dumps(header, ensure_ascii=False) + "\n")

    def log_correction(self, original, corrected, error_type, confidence, semantic_sim):
        # Save a correction to log file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "original": original,
            "corrected": corrected,
            "error_type": error_type,
            "confidence": float(confidence),
            "semantic_similarity": float(semantic_sim),
            "changed": original.strip() != corrected.strip(),
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def load_dataset(self):
        # Load MERLIN dataset 
        raw = load_dataset("symeneses/merlin", "german", trust_remote_code=True)

        def to_pair(example):
            return {
                "incorrect": str(example["text"]).strip(),
                "correct": str(example["text_target"]).strip(),
                "type": "grammatical_error",
            }

        # Apply transformation to all train examples
        cleaned_examples = [to_pair(ex) for ex in raw["train"]]
        total_examples = len(cleaned_examples)

        # 80/20 train/validation split
        train_size = int(0.8 * total_examples)
        train_data = cleaned_examples[:train_size]
        val_data = cleaned_examples[train_size:]

        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data)
                if val_data
                else Dataset.from_list(train_data[:2]),
            }
        )

        return dataset

    def setup_model(self):
        # Configure model with LoRA
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            # For T5 generally <pad> already exists
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Set consistent pad_token_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Conservative LoRA configuration
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q", "v"],  # Only query and value
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Move to device BEFORE training
        self.model = self.model.to(self.device)

        # Disable DataParallel explicitly
        if hasattr(self.model, "module"):
            self.model = self.model.module

    def preprocess_data(self, dataset):
        # Preprocess data for training
        def preprocess_function(batch):
            inputs = [f"Korrigiere: {text}" for text in batch["incorrect"]]
            targets = batch["correct"]

            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=48,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            # Tokenize targets with modern API
            labels = self.tokenizer(
                text_target=targets,
                max_length=48,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            # Replace pad tokens in labels with -100
            pad_id = self.tokenizer.pad_token_id
            labels_cleaned = [
                [-100 if t == pad_id else t for t in seq] for seq in labels["input_ids"]
            ]

            model_inputs["labels"] = labels_cleaned
            return model_inputs

        return dataset.map(
            preprocess_function,
            batched=True,
            batch_size=64,  # mapping batch size, not training
            remove_columns=dataset["train"].column_names,
        )

    def train_model(self, dataset, output_dir="./t5-single-gpu"):
        # Training 
        tokenized_dataset = self.preprocess_data(dataset)

        # Training arguments optimized for single GPU
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=10,
            learning_rate=5e-4,
            warmup_steps=200,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=48,
            load_best_model_at_end=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            # Disable DataParallel
            local_rank=-1,
            ddp_backend=None,
            report_to=None,  # Disable wandb
        )

        # Trainer 
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
        )

        # Ensure trainer doesn't use DataParallel
        trainer.model = self.model

        try:
            # Start training on single GPU
            trainer.train()

            # Save model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Optional evaluation with BLEU/ROUGE
            preds = trainer.predict(tokenized_dataset["validation"])
            decoded_preds = self.tokenizer.batch_decode(
                preds.predictions, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                [[t if t != -100 else self.tokenizer.pad_token_id for t in seq] for seq in preds.label_ids],
                skip_special_tokens=True,
            )
            bleu_score = self.bleu.compute(
                predictions=decoded_preds, references=[[l] for l in decoded_labels]
            )
            rouge_score = self.rouge.compute(
                predictions=decoded_preds, references=decoded_labels
            )

            return True

        except Exception as e:
            return False

    def setup_pipeline(self, model_path=None):
        # Configure pipeline
        try:
            device_id = 0 if self.device.type == "cuda" else -1

            if model_path and os.path.exists(model_path):
                self.gec_pipeline = pipeline(
                    "text2text-generation",
                    model=model_path,
                    tokenizer=model_path,
                    device=device_id,
                    max_length=48,
                    num_beams=2,
                )
            else:
                self.gec_pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device_id,
                    max_length=48,
                    num_beams=2,
                )

        except Exception as e:
            # Complete fallback to base model CPU
            self.gec_pipeline = pipeline(
                "text2text-generation", model="t5-small", device=-1, max_length=48
            )

    def correct_sentence(self, sentence):
        # Correct a sentence
        if not self.gec_pipeline:
            return sentence

        try:
            # Add prefix consistent with training
            prompt = f"Korrigiere: {sentence}"
            result = self.gec_pipeline(prompt, max_length=48)
            return result[0]["generated_text"]
        except Exception as e:
            return sentence

    def process_sentence(self, sentence):
        # Complete pipeline for a sentence
        corrected = self.correct_sentence(sentence)

        result = {
            "original": sentence,
            "corrected": corrected,
            "error_type": "grammatical_error",
            "error_confidence": 0.7,
            "semantic_similarity": 0.9,
            "changed": sentence.strip() != corrected.strip(),
        }

        # Save log
        self.log_correction(sentence, corrected, "grammatical_error", 0.7, 0.9)
        return result


def main():
    # Initialize corrector
    corrector = GermanCorrector()

    try:
        # Load dataset 
        dataset = corrector.load_dataset()

        # Setup model
        corrector.setup_model()

        # Training
        success = corrector.train_model(dataset)

        if success:
            corrector.setup_pipeline("./t5-single-gpu")
        else:
            corrector.setup_pipeline()

    except Exception as e:
        corrector.setup_pipeline()

    # Test sentences
    test_sentences = [
        "Ich habe gestern ins Kino gegangen .",
        "Der Hund laufen schnell .",
        "Sie haben viele Bücher gelest .",
        "Die Kinder spielen in der Park .",
        "Er geht zu die Schule jeden Tag .",
        "Ich habe meine Hausaufgaben nicht gemacht ,  weil ich war müde .",
        "Morgen ich will gehen in Kino .",
        "Die Auto ist sehr schnell .",
        "Wir haben gegessen Pizza gestern .",
        "Er spielt Fussball mit seine Freunde ."
    ]

    # Test corrections
    for i, sentence in enumerate(test_sentences, 1):
        try:
            result = corrector.process_sentence(sentence)
        except Exception as e:
            pass


if __name__ == "__main__":
    main()