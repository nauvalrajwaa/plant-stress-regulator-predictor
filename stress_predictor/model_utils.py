import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ['WANDB_DISABLED'] = 'true'
os.environ["USE_TRITON"] = "0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.models.bert.configuration_bert import BertConfig

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from transformers import logging
logging.set_verbosity_error()
use_cuda = torch.cuda.is_available()

import warnings
warnings.filterwarnings("ignore", message="Unable to import Triton")

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_tokenizer(tokenizer_name: str, is_dnabert: bool):
    tokenizer = AutoTokenizer.from_pretrained(f"igemugm/{tokenizer_name}-stress-predictor")
    tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        if is_dnabert:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    if 'token_type_ids' in tokenizer.model_input_names:
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != 'token_type_ids']

    return tokenizer

def prepare_model(model_name: str, config, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        f"igemugm/{model_name}-stress-predictor",
        trust_remote_code=True,
        config=config,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))
    return model

def load_model(model_name: str, tokenizer_name: str, device):
    if model_name != tokenizer_name:
        raise ValueError("Model name and tokenizer name must be the same")

    is_dnabert = "dnabert" in model_name.lower()

    if is_dnabert:
        config = BertConfig.from_pretrained(
            f"igemugm/{model_name}-stress-predictor",
            trust_remote_code=True
        )
    else:
        config = AutoConfig.from_pretrained(
            f"igemugm/{model_name}-stress-predictor",
            trust_remote_code=True
        )

    tokenizer = prepare_tokenizer(tokenizer_name, is_dnabert)
    config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model(model_name, config, tokenizer)

    return tokenizer, model


def tokenize_function(tokenizer, examples):
    result = tokenizer(
        examples,
        padding=False,
        truncation=True,
        max_length=200
    )
    return result

class SingleSequenceDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.data.items()}

def region_stress_classification(
    model_name, tokenizer_name, sequence, device, 
    window_size=200, stride=100, save_path="visualization.png",
    output_dir="outputs_rg"
): 
    seq_len = len(sequence)
    if seq_len != 1000 and seq_len != 2000:
        raise ValueError("Sequence length must be either 1000 or 2000")
    
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model= load_model(model_name, tokenizer_name, device)
    
    pos_votes = [[] for _ in range(seq_len)]
    results = {}
    all_probs = []

    window_id = 1
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        subseq = sequence[start:end]

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments( 
            output_dir="./results",   
            warmup_steps=0,                
            weight_decay=0.005, 
            learning_rate=2e-5,
            dataloader_pin_memory=use_cuda
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
        )

        tokenized = tokenize_function(tokenizer, subseq)
        dataset = SingleSequenceDataset(tokenized)
        test_predictions = trainer.predict(dataset)
        pred = np.argmax(test_predictions.predictions[0])
        if "dnabert" in model_name.lower():
            probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions[0]), dim=-1).cpu().tolist()[0]
        elif model_name.lower() == "mistral-athaliana":
            probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions[0]), dim=-1).cpu().tolist()

        results[str(window_id)] = {
            "sequence": subseq,
            "label_seq": str(pred),
            "score": probs[pred]
        }
        all_probs.append(probs[pred])

        for i in range(start, end):
            pos_votes[i].append(pred)

        window_id += 1

    results["final_score"] = sum(all_probs) / len(all_probs)

    # visualization
    colors = []
    for votes in pos_votes:
        if len(votes) == 0:
            colors.append("white")
        elif all(v == 1 for v in votes):
            colors.append("green")
        elif all(v == 0 for v in votes):
            colors.append("red")
        else:
            colors.append("gray")

    plt.figure(figsize=(15, 2))
    plt.scatter(range(seq_len), [1]*seq_len, c=colors, s=30, marker="s")
    plt.title("Region Stress Classification Visualization")
    plt.yticks([])
    plt.xlabel("Sequence Position")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.savefig(f"{output_dir}/{save_path}", dpi=300, bbox_inches="tight")
    plt.close()

    return results

def promoter_stress_classification(
    model_name, tokenizer_name, sequence, device, 
    slice_size=1000, stride=200, window_size=200, output_dir="outputs_pr"
):
    seq_len = len(sequence)
    if seq_len < 5000 or seq_len > 10000:
        raise ValueError("Sequence length must be between 5000 - 10000")
    elif seq_len % 1000 != 0:
        raise ValueError("Sequence length must be divisible by 1000")

    if slice_size == 1000:
        valid_strides = [100, 200, 500]
    elif slice_size == 2000:
        valid_strides = [100, 200, 400, 500]
    else:
        raise ValueError("Slice size must be either 1000 or 2000")

    if stride not in valid_strides:
        raise ValueError(f"Stride {stride} is not valid for slice {slice_size}. "
                         f"Valid: {valid_strides}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    num_slices = (len(sequence) + slice_size - 1) // slice_size

    for slice_id in range(num_slices):
        start = slice_id * slice_size
        end = min((slice_id + 1) * slice_size, len(sequence))
        subseq = sequence[start:end]

        save_path = f"slice{slice_id+1}_stride{stride}.png"
        res = region_stress_classification(
            model_name, tokenizer_name, subseq, device, 
            window_size=window_size, stride=stride, save_path=save_path,
            output_dir=output_dir
        )
        results[f"slice_{slice_id+1}"] = res

    return results
