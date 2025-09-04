# proj3

A deep learning project implementing NLP experiments, including **feature extraction with classifiers**, **transformer fine-tuning**, and **German grammatical error correction**.

> **NB:** All experiments in this repository are explained here in the initial `README.md`. Each script can be run independently.

<details>
<summary>🎯 Objectives</summary>

The exercises are designed to explore different NLP tasks and modeling strategies:

- **Feature Extraction + Baseline Classifiers**
  - Script: main_fine_tuning.py
  - Extract embeddings from pre-trained transformers
  - Train **SVM** and **Random Forest** classifiers on Rotten Tomatoes
  - Evaluate classification performance (accuracy, precision, recall, F1)

- **Fine-tuning Transformer (DistilBERT)**
  - Script: main_fine_tuning.py
  - Fine-tune **DistilBERT** for binary sentiment classification
  - Evaluate on validation and test sets
  - Save fine-tuned models for further use

- **German Grammatical Error Correction (GEC)**
  - Script: german.py
  - Train **T5-small with LoRA adapters** on MERLIN dataset (German)
  - Implement correction pipeline for sentences
  - Log corrections with:
    - Original sentence
    - Corrected version
    - Error type
    - Confidence score
    - Semantic similarity
    - Whether a change was applied
</details>

<details>
<summary>📂 Project Structure</summary>

proj3/
│── utils/                 # Utility scripts
│   ├── data_utils.py      # Dataset loading and preprocessing
│   ├── model_utils.py     # Model loading and feature extraction
│   ├── train_utils.py     # Training helpers (SVM, RF, metrics, etc.)
│
│── corrections.jsonl      # Logs of German grammar corrections
│── german.py              # German grammar correction with LoRA + T5
│── main_extract_f.py      # Feature extraction + baseline classifiers (SVM / RF)
│── main_fine_tuning.py    # Fine-tuning DistilBERT on Rotten Tomatoes
│── todo.txt               # Notes and pending tasks
│── requirements.txt       # Project dependencies

</details>
