
import os
import csv
import json
import logging
import shutil
import glob
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

# Attempt to import PEFT
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] PEFT not installed. LoRA will be unavailable.")

# =============================================================================
# 1. PATCHING UTILITIES (From Notebook Cells 5-6)
# =============================================================================
def fix_dnabert2_layers(local_model_path):
    """
    Applies the 'Safe Patch' to bert_layers.py to prevent Flash Attention/Triton errors.
    """
    bert_layers_path = os.path.join(local_model_path, "bert_layers.py")
    if not os.path.exists(bert_layers_path):
        print(f"[WARN] {bert_layers_path} not found. Skipping patch.")
        return

    print(f"[INFO] Patching {bert_layers_path} for Flash Attention compatibility...")
    
    with open(bert_layers_path, "r") as f:
        lines = f.readlines()

    with open(bert_layers_path, "w") as f:
        for line in lines:
            # Replace dangerous import with None assignment
            if "from .flash_attn_triton import" in line:
                indent = line[:len(line) - len(line.lstrip())]
                f.write(f"{indent}flash_attn_qkvpacked_func = None\n")
                f.write(f"{indent}flash_attn_func = None\n")
                # print("       -> Replaced flash_attn_triton import with None.")
            else:
                f.write(line)
    
    # Clear HF Cache if possible (optional, but good practice)
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.exists(cache_dir):
        try:
            # shutil.rmtree(cache_dir) 
            # Be careful deleting global cache automatically.
            pass
        except:
            pass
    print("[INFO] Patch Applied.")

def ensure_model_code_files(source_dir, target_dir):
    """
    Copies all .py files from source model dir to output dir to ensure
    custom model architecture code travels with weights.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    py_files = glob.glob(os.path.join(source_dir, "*.py"))
    count = 0
    for file_path in py_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(target_dir, file_name)
        shutil.copy2(file_path, dest_path)
        count += 1
    print(f"[INFO] Copied {count} python files to {target_dir} for reproducibility.")

# =============================================================================
# 2. DATASET CLASSES (From Notebook Cell 4)
# =============================================================================
def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        kmer = [generate_kmer_str(text, k) for text in texts]
        # Only save if we have a valid path (sometimes temp files don't need saving)
        if data_path and not data_path.endswith("tmp"): 
             with open(kmer_path, "w") as f:
                json.dump(kmer, f)
    return kmer

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        
        # Read CSV
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:] # Skip header
            
        if not data:
            raise ValueError(f"Data file {data_path} is empty or invalid.")

        # Parse Columns
        if len(data[0]) >= 2:
            texts = [d[0] for d in data]
            labels = [int(float(d[1])) for d in data] # Handle potential float strings
        else:
            raise ValueError("Data format not supported. Expecting [sequence, label]")

        if kmer != -1:
            print(f"[INFO] Using {kmer}-mer input format...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

        # Tokenize
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# =============================================================================
# 3. METRICS
# =============================================================================
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)


# =============================================================================
# 4. MAIN WRAPPER FUNCTION
# =============================================================================
def run_dnabert2_finetuning(
    train_csv_path: str,
    val_csv_path: str,
    output_dir: str,
    model_name_or_path: str = "zhihan1996/DNABERT-2-117M",
    epochs: int = 3,
    batch_size: int = 8,
    save_model: bool = True,
    use_lora: bool = False
):
    """
    Main entry point for DNABERT-2 finetuning.
    Compatible with integration into other scripts.
    """
    print(f"\n[DNABERT-2] Starting Finetuning Pipeline...")
    print(f"Model: {model_name_or_path}")
    print(f"Data: {train_csv_path}")
    print(f"Output: {output_dir}")

    # 1. FIX LAYERS (Safe Patch)
    if os.path.exists(model_name_or_path):
        fix_dnabert2_layers(model_name_or_path)
    
    # 2. SETUP TOKENIZER
    # trust_remote_code=True is essential
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. PREPARE DATASETS
    # Note: notebook used separate CSVs. We expect paths passed in.
    print("[DNABERT-2] Loading Datasets...")
    train_dataset = SupervisedDataset(train_csv_path, tokenizer=tokenizer, kmer=-1)
    val_dataset = SupervisedDataset(val_csv_path, tokenizer=tokenizer, kmer=-1)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # 4. LOAD MODEL WITH EXPLICIT CONFIG (Fix Config Mismatch)
    print("[DNABERT-2] Loading Model with explicit config...")
    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    
    # 5. LORA (Optional)
    if use_lora and PEFT_AVAILABLE:
        print("[DNABERT-2] Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 6. TRAINING ARGS
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none"
    )

    # 7. TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    print("[DNABERT-2] Training...")
    trainer.train()

    # 8. SAVE
    if save_model:
        print(f"[DNABERT-2] Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Copy custom code files for portability
        if os.path.isdir(model_name_or_path):
             ensure_model_code_files(model_name_or_path, output_dir)
        else:
             # If downloaded from HF, we might need to find where it is cached or trust that save_pretrained did it.
             # save_pretrained usually saves the custom code if trust_remote_code was user.
             pass

    return model, tokenizer

# Simple CLI for testing independently
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", default="dnabert2_output")
    args = parser.parse_args()
    
    run_dnabert2_finetuning(args.train_csv, args.val_csv, args.output_dir)
