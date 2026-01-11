import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ['WANDB_DISABLED'] = 'true'
os.environ["USE_TRITON"] = "0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers.models.bert.configuration_bert import BertConfig

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F

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
    """Tokenize sequence for model input."""
    result = tokenizer(
        examples,
        padding=False,
        truncation=True,
        max_length=200
    )
    return result


def get_prediction_probabilities(model, tokenizer, sequence, device):
    """
    Get prediction probabilities using direct model inference.
    
    Args:
        model: The pre-loaded model
        tokenizer: The tokenizer
        sequence: DNA sequence string
        device: torch device (cuda or cpu)
    
    Returns:
        tuple: (predicted_label, probabilities_list, confidence_score)
    """
    # Tokenize input
    inputs = tokenizer(
        sequence,
        padding=True,
        truncation=True,
        max_length=200,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Calculate probabilities
    probs = F.softmax(logits, dim=-1)
    probs_list = probs.cpu().squeeze().tolist()
    
    # Handle single vs multiple class outputs
    if not isinstance(probs_list, list):
        probs_list = [probs_list]
    
    # Get prediction
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs_list[pred]
    
    return pred, probs_list, confidence


def region_stress_classification(
    model_name, tokenizer_name, sequence, device, 
    window_size=200, stride=100, save_path="visualization.png",
    output_dir="outputs_rg"
): 
    """
    Classify stress regions in a DNA sequence using a sliding window approach.
    
    Args:
        model_name: Name of the model to use (dnabert or mistral-athaliana)
        tokenizer_name: Name of the tokenizer (should match model_name)
        sequence: DNA sequence string (must be 1000 or 2000 bp)
        device: torch device for computation
        window_size: Size of sliding window (default: 200)
        stride: Step size for sliding window (default: 100)
        save_path: Filename for visualization output
        output_dir: Directory to save outputs
    
    Returns:
        dict: Results containing predictions for each window and final score
    """
    seq_len = len(sequence)
    if seq_len != 1000 and seq_len != 2000:
        raise ValueError("Sequence length must be either 1000 or 2000")
    
    os.makedirs(output_dir, exist_ok=True)

    # Load model ONCE before the loop (CRITICAL PERFORMANCE FIX)
    try:
        tokenizer, model = load_model(model_name, tokenizer_name, device)
        model.to(device)
        model.eval()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA out of memory! Falling back to CPU...")
            device = torch.device("cpu")
            tokenizer, model = load_model(model_name, tokenizer_name, device)
            model.to(device)
            model.eval()
        else:
            raise
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
    
    pos_votes = [[] for _ in range(seq_len)]
    results = {}
    all_probs = []

    window_id = 1
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        subseq = sequence[start:end]

        # Use direct inference instead of Trainer (PERFORMANCE FIX)
        try:
            pred, probs, confidence = get_prediction_probabilities(
                model, tokenizer, subseq, device
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA OOM at window {window_id}. Clearing cache...")
                torch.cuda.empty_cache()
                pred, probs, confidence = get_prediction_probabilities(
                    model, tokenizer, subseq, device
                )
            else:
                raise

        results[str(window_id)] = {
            "sequence": subseq,
            "label_seq": str(pred),
            "score": confidence
        }
        all_probs.append(confidence)

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
    """
    Classify stress promoters in a DNA sequence by splitting into slices.
    
    This function divides long sequences (5000-10000 bp) into smaller slices
    and performs region stress classification on each slice.
    
    Args:
        model_name: Name of the model to use (dnabert or mistral-athaliana)
        tokenizer_name: Name of the tokenizer (should match model_name)
        sequence: DNA sequence string (must be 5000-10000 bp, divisible by 1000)
        device: torch device for computation
        slice_size: Size of each slice (1000 or 2000, default: 1000)
        stride: Step size for sliding window within slices (default: 200)
        window_size: Size of sliding window within slices (default: 200)
        output_dir: Directory to save outputs
    
    Returns:
        dict: Results containing predictions for each slice
    """
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
        
        # region_stress_classification now has model loading optimized
        res = region_stress_classification(
            model_name, tokenizer_name, subseq, device, 
            window_size=window_size, stride=stride, save_path=save_path,
            output_dir=output_dir
        )
        results[f"slice_{slice_id+1}"] = res

    return results
