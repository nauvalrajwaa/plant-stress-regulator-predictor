
import os
import csv
import json
import logging
import random
import sys
import numpy as np
import pandas as pd
import torch
import transformers
import sklearn
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from torch.utils.data import Dataset
from transformers import BertConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# 0. SYSTEM OPTIMIZATIONS (Preventing Errors in Colab/Legacy Envs)
# =============================================================================

def fix_libtinfo():
    """
    Attempts to fix the missing libtinfo.so.5 error common in Colab/newer Linux
    when running older binaries or libraries (like DNABERT dependencies).
    """
    try:
        if os.path.exists("/usr/lib/x86_64-linux-gnu/libtinfo.so.6"):
            if not os.path.exists("/usr/lib/x86_64-linux-gnu/libtinfo.so.5"):
                print("[System] Applying libtinfo.so.5 symlink fix...")
                os.system("ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /usr/lib/x86_64-linux-gnu/libtinfo.so.5")
            
            if not os.path.exists("/usr/lib/x86_64-linux-gnu/libtinfo.so"):
                 os.system("ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /usr/lib/x86_64-linux-gnu/libtinfo.so")
    except Exception as e:
        print(f"[System] Warning: Could not apply libtinfo fix (Permission denied?): {e}")

# =============================================================================
# 1. K-MER UTILITIES (Core to DNABERT-1)
# =============================================================================

def seq2kmer(seq: str, k: int) -> str:
    """
    Converts a DNA sequence into a space-separated k-mer string.
    Example (k=3): "ACGT" -> "ACG CGT"
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    return " ".join(kmer)

def load_and_process_data(data_path: str, k: int) -> Tuple[List[str], List[int]]:
    """
    Loads CSV [sequence, label] and converts sequences to k-mers.
    """
    sequences = []
    labels = []
    
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header
        
        for row in reader:
            if len(row) >= 2:
                seq = row[0].upper().strip()
                label = int(float(row[1]))
                
                # Convert to K-mer string
                kmer_str = seq2kmer(seq, k)
                if kmer_str: # Ensure not empty
                    sequences.append(kmer_str)
                    labels.append(label)
                    
    return sequences, labels

# =============================================================================
# 2. DATASET CLASS
# =============================================================================

class DNABERT1Dataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], tokenizer: transformers.PreTrainedTokenizer, max_len: int = 512):
        self.len = len(sequences)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = sequences

    def __getitem__(self, index):
        seq = self.sequences[index]
        label = self.labels[index]

        # DNABERT-1 uses BERT tokenizer. Input is "ACG CGT ...".
        # We assume tokenizer is configured correctly.
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len

# =============================================================================
# 3. METRICS
# =============================================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = sklearn.metrics.accuracy_score(labels, predictions)
    f1 = sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0)
    precision = sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0)
    recall = sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0)
    mcc = sklearn.metrics.matthews_corrcoef(labels, predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc
    }

# =============================================================================
# 4. MAIN WRAPPER FUNCTION
# =============================================================================

def run_dnabert1_finetuning(
    train_csv_path: str,
    val_csv_path: str,
    output_dir: str,
    kmer: int = 6,
    epochs: int = 3,
    batch_size: int = 16,
    save_model: bool = True,
    model_name_or_path: str = None
):
    """
    Main entry point for DNABERT-1 finetuning.
    Adapts the notebook logic to modern Transformers to run in the same pipeline.
    """
    # [Optimization] Apply system fixes (libtinfo) if needed
    fix_libtinfo()

    # 1. Determine Model Path
    # DNABERT-1 models are usually named DNA_bert_3, DNA_bert_6 etc.
    if not model_name_or_path:
        model_name_or_path = f"zhihan1996/DNA_bert_{kmer}"
    
    print(f"\n[DNABERT-1] Starting Finetuning Pipeline...")
    print(f"Model ID: {model_name_or_path}")
    print(f"K-mer Size: {kmer}")
    print(f"Data: {train_csv_path}")

    # 2. Setup Tokenizer
    # DNABERT-1 uses standard BERT tokenizer
    print("[DNABERT-1] Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {model_name_or_path}. Fallback to bert-base-uncased.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3. Prepare Data
    print("[DNABERT-1] Processing Data (generating k-mers)...")
    train_texts, train_labels = load_and_process_data(train_csv_path, kmer)
    val_texts, val_labels = load_and_process_data(val_csv_path, kmer)
    
    train_dataset = DNABERT1Dataset(train_texts, train_labels, tokenizer, max_len=512)
    val_dataset = DNABERT1Dataset(val_texts, val_labels, tokenizer, max_len=512)
    
    print(f"      -> Train size: {len(train_dataset)}")
    print(f"      -> Val size: {len(val_dataset)}")

    # 4. Load Model
    print("[DNABERT-1] Loading Model...")
    # Determine number of labels
    num_labels = len(set(train_labels))
    
    # FIX: Explicitly handle config loading for missing HF models
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True,
            hidden_dropout_prob=0.1
        )
    except OSError:
        # If zhihan1996/DNA_bert_6 fails (common issue), use specific config
        print(f"[WARN] Could not load config directly from {model_name_or_path}.")
        print("       --> Attempting to download manually or use fallback config.")
        
        # Fallback: Download manually if not present
        if not os.path.exists(model_name_or_path) and "zhihan1996" in model_name_or_path:
             os.system(f"git clone https://huggingface.co/{model_name_or_path} {model_name_or_path}")
        
        # Retry load
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True
        )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-4, # Higher LR for DNABERT-1 as per paper/notebook
        weight_decay=0.01,
        warmup_ratio=0.1,   # Match notebook warmup_percent 0.1
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none"
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("[DNABERT-1] Training...")
    trainer.train()

    # 7. Save
    if save_model:
        print(f"[DNABERT-1] Saving to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save config kmer for inference later
        config_path = os.path.join(output_dir, "dnabert_config.json")
        with open(config_path, "w") as f:
            json.dump({"kmer": kmer, "model_type": "dnabert1"}, f)

    return model, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", default="dnabert1_output")
    parser.add_argument("--kmer", type=int, default=6)
    args = parser.parse_args()
    
    run_dnabert1_finetuning(args.train_csv, args.val_csv, args.output_dir, args.kmer)
