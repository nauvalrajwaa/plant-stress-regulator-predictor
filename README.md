# Stress Predictor & Training Pipeline

A comprehensive toolkit for predicting, analyzing, and training models for stress-responsive regions in plant DNA. This repository contains both the **Inference Engine** (for making predictions) and the **Training Pipeline** (for mining data from NCBI and training new custom models).

## 🌟 Features

*   **Dual Mode Inference**: 
    *   `--rg`: Small region analysis (1-2kb)
    *   `--pr`: Promoter scanning (up to 10kb) with adaptive slicing.
*   **End-to-End Training**: Automated pipeline to **Search** genes, **Mine** sequences, **Check** validity, and **Train** ensemble models.
*   **Visual Analytics**: 
    *   Probability Heatmaps (Green = Stress, Red = Non-Stress).
    *   Sequence Logo generation for motif analysis.
*   **Dynamic Configuration**: Flexible support for different organisms (*Arabidopsis*, *Rice*, etc.) and gene limits.

---

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/venusangela/stress-predictor.git
    cd stress-predictor
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔍 Part 1: Inference (Running Predictions)

Use `stress_predictor/main.py` to predict stress regions on your FASTA files.

### 📋 Full Command Reference

| Flag | Required | Description | Default |
| :--- | :---: | :--- | :--- |
| `--input` | ✅ | Path to the input FASTA file containing sequences. | - |
| `--pr` | *One of* | **Promoter Mode**: Scans long sequences (e.g., 2kb-10kb) using a sliding window. | - |
| `--rg` | *One of* | **Region Mode**: Classifies short DNA chunks (e.g., 200bp) directly. | - |
| `--model-path` | ❌ | Path to a local model folder (e.g., `runs/run_ID/models/...`). **Overrides --model**. | `None` |
| `--model` | ❌ | Hugging Face Hub model ID (if not using local path). | `None` |
| `--tokenizer` | ❌ | Hugging Face Hub tokenizer ID (usually same as model). | `None` |
| `--output` | ❌ | Custom folder to save results. | `runs/run_{date}_{mode}` |
| `--slice` | ❌ | Window size (base pairs) for promoter scanning (`--pr` only). | `1000` |
| `--stride` | ❌ | Step size for sliding window (`--pr` only). | `200` |
| `--force-cpu` | ❌ | Force CPU execution even if GPU (CUDA/MPS) is available. | `False` |

### Examples

**1. Promoter Mode (Scanning Long Sequences):**
```bash
python stress_predictor/main.py \
    --input "data/promoter_sequences.fasta" \
    --pr \
    --model-path "runs/run_20260113_Rice_agront/models/PlantBERT_98.5_Oryza_sativa"
```

**2. Region Mode (Quick Classification):**
```bash
python stress_predictor/main.py \
    --input "data/short_regions.fasta" \
    --rg \
    --model "nigelhartm/PlantBERT" \
    --tokenizer "nigelhartm/PlantBERT"
```

---

## 🛠 Part 2: Training Pipeline

Use `scripts/train.py` to build datasets from NCBI and fine-tune models (PlantBERT, DNABERT-2, Agro-NT).

### 📋 Full Command Reference

| Flag | Description | Default |
| :--- | :--- | :--- |
| **Pipeline Control** | | |
| `--step` | Steps to execute: `all` (Full), `search` (NCBI), `mine` (Sequences), `train` (Fine-tune), `eval`. | `all` |
| `--email` | **[Required]** Email address (for NCBI Entrez usage). | `user@example.com` |
| **Model Configuration** | | |
| `--llm-model` | **[New]** Choose Foundation Model: `plantbert`, `dnabert2`, or `agront` (1B params). | `plantbert` |
| `--save-models` | Save valid model checkpoints to `runs/`. Use `--no-save-models` to disable. | `True` |
| `--model-path` | Path to a specific base model (local or HF) if not using presets. | `None` |
| **Data Mining** | | |
| `--organism` | Target scientific name (e.g., "Oryza sativa", "Zea mays"). | "Arabidopsis thaliana" |
| `--limit-genes` | Max number of genes to fetch from NCBI. Set `0` for no limit. | `100` |
| `--task-type` | `binary` (Stress vs Random) or `multiclass` (Stress Type). | `binary` |
| `--max-seq-len` | Maximum length of sequences to download to avoid memory issues. | `20000` |
| `--flank-bp` | Number of base pairs to extract around a detected motif. | `50` |
| **File Paths (Optional)** | | |
| `--gene-list` | Path to a custom .txt file of gene names/accessions. | Auto-generated |
| `--mined-data` | Path to a custom .csv file for training. | Auto-generated |
| `--place-csv` | Path to PLACE database CSV. | Auto-detected |

### Example Commands

**1. Train with Agro-NT (1 Billion Parameters):**
*Automatically handles memory optimization (batch size=1, grad_accum=16).*
```bash
python scripts/train.py \
    --step all \
    --organism "Nicotiana tabacum" \
    --llm-model agront \
    --limit-genes 200
```

**2. Train with DNABERT-2 (Long Context):**
*Uses 1024bp context window and custom ALiBi attention.*
```bash
python scripts/train.py \
    --step all \
    --organism "Zea mays" \
    --llm-model dnabert2 \
    --limit-genes 500
```

**3. Dataset Creation Only (No Training):**
```bash
python scripts/train.py \
    --step mine \
    --organism "Triticum aestivum" \
    --limit-genes 1000
```

---

## 📊 Part 3: Data Analysis

Use `scripts/analyze_results.py` for deeper inspection of your datasets *before* or *after* training.

```bash
# Auto-detects the last run for this config
python scripts/analyze_results.py --organism "Oryza sativa" --size 500

# Explicit file analysis
python scripts/analyze_results.py --input "datasets/dataset_custom.csv" --mode stats
```

**Analysis Modes (`--mode`):**
*   `stats`: Show length distribution, class imbalance, and duplicates.
*   `check`: Validate IUPAC characters in DNA sequences.
*   `logo`: Generate sequence logo images in `analysis/` folder.

---

## 📂 Folder Structure

```text
stress-predictor/
├── datasets/                 # 📂 Stores mined CSVs and gene lists (Auto-generated)
├── plots/                    # 📂 Stores Sequence Logos (png)
├── runs/                     # 📂 Stores Inference outputs (Heatmaps, HTML, JSON)
├── scripts/
│   ├── train.py              # 🧠 Main Training Pipeline
│   ├── analyze_results.py    # 📊 Data Analysis Tool
│   └── ensemble_predictor.py # 🤖 Model Definitions
├── stress_predictor/
│   ├── main.py               # 🔮 Inference Entry Point
│   └── model_utils.py        # ⚙️ Inference Logic
└── software_test/            # 🧪 Test FASTA files
```

---
*Created for Stress Region Prediction Project*
