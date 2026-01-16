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

def prepare_tokenizer(tokenizer_name_or_path: str, is_dnabert: bool, is_local=False):
    path = tokenizer_name_or_path if is_local else f"igemugm/{tokenizer_name_or_path}-stress-predictor"
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        if is_dnabert:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models like PlantBERT/BERT that might miss pad_token config
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if 'token_type_ids' in tokenizer.model_input_names:
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != 'token_type_ids']

    return tokenizer

def prepare_model(model_name_or_path: str, config, tokenizer, is_local=False):
    path = model_name_or_path if is_local else f"igemugm/{model_name_or_path}-stress-predictor"
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        trust_remote_code=True,
        config=config,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))
    return model

def load_model(model_name: str, tokenizer_name: str, device, model_path: str = None):
    if model_path is None and (not model_name or not tokenizer_name):
        raise ValueError("Model name and tokenizer name required if model_path is missing")
    
    if model_path is None and model_name != tokenizer_name:
        raise ValueError("Model name and tokenizer name must be the same")

    is_dnabert = False
    if model_name:
        is_dnabert = "dnabert" in model_name.lower()
    elif model_path:
        is_dnabert = "dnabert" in model_path.lower()
    
    # Determine the actual path or name to use
    target_path = model_path if model_path else model_name
    is_local = (model_path is not None)

    if is_local:
         if is_dnabert:
             config = BertConfig.from_pretrained(target_path, trust_remote_code=True)
         else:
             config = AutoConfig.from_pretrained(target_path, trust_remote_code=True)
    else:
         if is_dnabert:
            config = BertConfig.from_pretrained(f"igemugm/{model_name}-stress-predictor", trust_remote_code=True)
         else:
            config = AutoConfig.from_pretrained(f"igemugm/{model_name}-stress-predictor", trust_remote_code=True)

    tokenizer = prepare_tokenizer(target_path, is_dnabert, is_local=is_local)
    # config.pad_token_id = tokenizer.pad_token_id # Moved after model loading
    model = prepare_model(target_path, config, tokenizer, is_local=is_local)
    model.config.pad_token_id = tokenizer.pad_token_id

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
    # Determine max length safely
    max_len = 200
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        max_len = min(max_len, model.config.max_position_embeddings)

    # Tokenize input
    inputs = tokenizer(
        sequence,
        padding=True,
        truncation=True,
        max_length=max_len,
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
    output_dir="outputs_rg", model_path=None
): 
    """
    Classify stress regions using adaptive slicing and probability mapping.
    """
    seq_len = len(sequence)
    # Flexible length check
    if seq_len < 200:
        raise ValueError("Sequence length too short (< 200)")
    
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    try:
        tokenizer, model = load_model(model_name, tokenizer_name, device, model_path=model_path)
        model.to(device)
        model.eval()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA out of memory! Falling back to CPU...")
            device = torch.device("cpu")
            tokenizer, model = load_model(model_name, tokenizer_name, device, model_path=model_path)
            model.to(device)
            model.eval()
        else:
            raise
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # 1. Smart Window Adaptation
    try:
        max_model_len = model.config.max_position_embeddings
    except AttributeError:
        max_model_len = 512 # Fallback
        
    print(f"Model max length: {max_model_len}")
    
    # Logic:
    # 1. If window_size is None (Auto mode), pick optimal based on model.
    # 2. If window_size is provided but too large, cap it.
    
    if window_size is None:
        # Default Auto Logic
        # For huge models (e.g. 32k), we don't want 32k windows (too heavy, dilutes signal).
        # We cap at 512 for stability, or use max_len-2 if smaller.
        window_size = min(512, max_model_len - 2)
        # print(f"   [Auto-Adjust] Set window size to {window_size} bp based on model architecture.")
        
        # Auto-adjust stride too if not set (assumed passed as default 100/200 usually)
        # Optimal stride is usually 50% overlap
        stride = window_size // 2
        # print(f"   [Auto-Adjust] Set stride to {stride} bp.")

    elif max_model_len < window_size:
        print(f"   [Conflict] Requested window {window_size} > Model Max {max_model_len}.")
        window_size = max_model_len - 2 # -2 for [CLS] and [SEP]
        print(f"   [Adapting] Reduced window size to {window_size} bp.")
        # Adjust stride to keep ratio if needed
        if stride >= window_size:
            stride = window_size // 2
            print(f"   [Adapting] Reduced stride to {stride} bp.")

    # Validation
    if window_size < 10:
        raise ValueError(f"Resulting window size {window_size} is too small!")
        stride = window_size // 2 # 50% overlap
        print(f"New Window: {window_size}, New Stride: {stride}")

    # 2. Probability Accumulation (Heatmap approach)
    # We store list of probabilities for Class 1 (Stress) for each base position
    pos_probs = [[] for _ in range(seq_len)]
    all_window_confs = []

    window_id = 1
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        subseq = sequence[start:end]

        pred, probs_list, confidence = get_prediction_probabilities(
            model, tokenizer, subseq, device
        )
        
        # Extract Probability of Class 1 (Stress)
        # If binary: probs_list[1]
        # If only 1 value provided (e.g. sigmoid), handle accordingly.
        # Ensure probs_list is list of [prob_0, prob_1]
        stress_prob = probs_list[1] if len(probs_list) > 1 else probs_list[0]
        
        all_window_confs.append(confidence)

        for i in range(start, end):
            pos_probs[i].append(stress_prob)

        window_id += 1

    # 3. Calculate Average Probability per Position
    avg_probs = []
    for p_list in pos_probs:
        if len(p_list) == 0:
            avg_probs.append(0.0)
        else:
            avg_probs.append(sum(p_list) / len(p_list))

    final_score = sum(avg_probs) / len(avg_probs) if avg_probs else 0.0

    # 4. Region Extraction
    regions = []
    threshold = 0.5
    min_region_len = 50
    
    in_region = False
    reg_start = 0
    
    for i, p in enumerate(avg_probs):
        if p >= threshold:
            if not in_region:
                in_region = True
                reg_start = i
        else:
            if in_region:
                in_region = False
                if (i - reg_start) >= min_region_len:
                    # Calculate avg confidence for this region
                    reg_conf = sum(avg_probs[reg_start:i]) / (i - reg_start)
                    regions.append({
                        "start": reg_start,
                        "end": i,
                        "length": i - reg_start,
                        "avg_prob": float(f"{reg_conf:.4f}")
                    })
    # Check if region ends at sequence end
    if in_region and (seq_len - reg_start) >= min_region_len:
         reg_conf = sum(avg_probs[reg_start:seq_len]) / (seq_len - reg_start)
         regions.append({
             "start": reg_start,
             "end": seq_len,
             "length": seq_len - reg_start,
             "avg_prob": float(f"{reg_conf:.4f}")
         })

    results = {
        "final_score": final_score,
        "regions": regions,
        "raw_probs": avg_probs[::10] # Downsample for lighter JSON if needed
    }

    # 5. Advanced Visualization (Gradient Heatmap)
    # Map probabilities to colors: Red(0) -> Yellow(0.5) -> Green(1.0)
    cmap = plt.get_cmap("RdYlGn")
    
    plt.figure(figsize=(15, 3))
    
    # Scatter plot with color mapping (back to top position y=1)
    sc = plt.scatter(range(seq_len), [1]*seq_len, c=avg_probs, cmap=cmap, vmin=0, vmax=1, s=20, marker="|")
    
    # Plot smoothed curve
    # Moving average for cleaner curve
    window_avg = 20
    if len(avg_probs) > window_avg:
        smoothed = [sum(avg_probs[i:i+window_avg])/window_avg for i in range(len(avg_probs)-window_avg)]
        plt.plot(range(window_avg//2, len(smoothed)+window_avg//2), smoothed, color='black', alpha=0.5, linewidth=1, label="Smoothed Prob")

    # Highlight Regions
    for reg in regions:
        plt.hlines(y=1.05, xmin=reg['start'], xmax=reg['end'], colors='blue', linewidth=3)
        plt.text((reg['start']+reg['end'])/2, 1.1, f"{reg['avg_prob']:.2f}", ha='center', fontsize=8, color='blue')

    plt.title("Stress Region Probability Map")
    
    # Enable Y-Axis ticks 0.0 to 1.0 (The requested feature)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)
    plt.ylabel("Probability", fontsize=10)
    plt.ylim(-0.1, 1.2)
    
    plt.xlabel("Sequence Position (bp)")
    plt.colorbar(sc, label="Stress Probability", orientation="horizontal", pad=0.2)
    
    # Add a custom legend (Previous style)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='yellow', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=2)]
    
    plt.legend(custom_lines, ['High Stress (>0.8)', 'Uncertain (0.4-0.6)', 'Non-Stress (<0.2)', 'Detected Region'], 
               loc='upper right', fontsize='small')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200 if seq_len > 1000 else 100))
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    # Adjust layout to make room for colorbar and legend
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{save_path}", dpi=300, bbox_inches="tight")
    plt.close()

    if len(regions) > 0:
        print(f"   > Found {len(regions)} potentially stress-responsive regions:")
        # List top 5 descending by avg_prob
        sorted_regs = sorted(regions, key=lambda x: x['avg_prob'], reverse=True)
        for i, reg in enumerate(sorted_regs[:5]):
             print(f"     * Region {i+1}: {reg['start']}-{reg['end']} bp (Confidence: {reg['avg_prob']:.2f})")
        if len(regions) > 5:
            print(f"     * ... and {len(regions)-5} more.")
    else:
        print(f"   > No significant stress regions found (threshold > 0.5).")

    return results

def promoter_stress_classification(
    model_name, tokenizer_name, sequence, device,
    slice_size=1000, stride=200, window_size=200, output_dir="outputs_pr",
    model_path=None
):
    """
    Classify stress promoters in a DNA sequence by splitting into slices.
    Now supports FLEXIBLE slicing for any sequence length.
    
    Args:
        model_name: Name of the model
        tokenizer_name: Name of the tokenizer
        sequence: DNA sequence string (Any length > window_size)
        device: torch device
        slice_size: Size of each slice (default: 1000)
        stride: Step size for sliding window (default: 200)
    """
    seq_len = len(sequence)
    
    # Handle None window_size (Auto-detect mode)
    # If None, we enforce a minimum safe length (e.g. 200) for slicing
    check_size = window_size if window_size is not None else 200

    # 1. Validation (Relaxed)
    if seq_len < check_size:
        raise ValueError(f"Sequence length ({seq_len}) is smaller than required minimum ({check_size})")

    print(f"--- Promoter Scanning ---")
    print(f"Sequence Length: {seq_len} bp")
    print(f"Configuration: Slice={slice_size}, Stride={stride}, Window={window_size if window_size is not None else 'Auto'}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # 2. Flexible Slicing Loop WITH OVERLAP
    # Improved Logic: We now use an overlap between slices to prevent 'Boundary Blindness'.
    # If a motif is split exactly between slice 1 and 2, neither detects it.
    # Overlap ensures it sits safely in the middle of at least one slice.
    
    overlap = 200  # 200bp safety overlap (typical model window size)
    step = slice_size - overlap
    if step <= 0: step = slice_size // 2  # Safety for very small slice settings
    
    current_start = 0
    slice_id = 0
    
    while current_start < seq_len:
        slice_id += 1
        start = current_start
        end = min(start + slice_size, seq_len)
        
        # Stop if we are just re-processing a tiny tail that's already fully covered
        # (happens if step is small and we reach end)
        if start > 0 and (end - start) < check_size:
             # If this last chunk is tiny, it was likely covered by the previous slice's overlap
             break

        subseq = sequence[start:end]
        print(f"\nProcessing Slice {slice_id}: {start}-{end} bp (Overlap: {start - (current_start-step) if slice_id > 1 else 0} bp)")

        save_path = f"slice{slice_id}_{start}_{end}.png"
        
        try:
            res = region_stress_classification(
                model_name, tokenizer_name, subseq, device,
                window_size=window_size, stride=stride, save_path=save_path,
                output_dir=output_dir, model_path=model_path
            )
            results[f"slice_{slice_id}"] = res
        except Exception as e:
            print(f"   [ERROR] Failed on Slice {slice_id}: {e}")

        # Move to next slice
        if end == seq_len:
            break
        current_start += step

    # Final Summary for Promoter Scanning

    # Final Summary for Promoter Scanning
    total_found = 0
    all_regions = []
    for slice_key, res_dict in results.items():
        if 'regions' in res_dict:
            total_found += len(res_dict['regions'])
            all_regions.extend(res_dict['regions']) # Note: these coordinates are relative to the slice!
    
    print("-" * 40)
    print(f"Summary: Found {total_found} stress regions across {slice_id} slices.")
    
    return results
