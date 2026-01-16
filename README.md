# Stress Predictor & Training Pipeline

A comprehensive toolkit for predicting, analyzing, and training models for stress-responsive regions in plant DNA. This repository contains both the **Inference Engine** (for making predictions) and the **Training Pipeline** (for mining data from NCBI and training new custom models).

## 🌟 Features

*   **Dual Mode Inference**: 
    *   `--rg` (**Region Mode**): Rapid classification of short DNA chunks (ideal for <2kb).
    *   `--pr` (**Promoter Mode**): **Smart Overlap Scanning** for long sequences (up to 10kb+). Uses overlapping slices (automatically overlaps by 200bp) to eliminate "boundary blindness" and ensure motifs split between slices are detected.
*   **Enhanced Visualization & Reporting**:
    *   Generates interactive **HTML Reports** with detailed legends and embedded plots.
    *   **High-Quality Plots**: PNG visualizations with smoothed trend lines, explicit Y-axis probability values, and clear stress/non-stress indicators.
*   **End-to-End Training**: Automated pipeline to **Search** genes, **Mine** sequences, **Check** validity, and **Train** ensemble models.
*   **Biological Validity**: Uses **Genomic Background Mining** to extract true negative samples (non-stress regions from the same gene) instead of generating random synthetic noise.
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
| --pr | *One of* | **Promoter Mode**: Scans long sequences with **Smart Overlap**. Splits sequence into chunks of `--slice` size with 200bp overlap. | - |
| --rg | *One of* | **Region Mode**: Classifies short DNA chunks (e.g., 200bp) directly. | - |
| --model-path | ❌ | Path to a local model folder (e.g., `runs/run_ID/models/...`). **Overrides --model**. | `None` |
| --model | ❌ | Hugging Face Hub model ID (if not using local path). | `None` |
| --tokenizer | ❌ | Hugging Face Hub tokenizer ID (usually same as model). | `None` |
| --output | ❌ | Custom folder to save results. | `runs/run_{date}_{mode}` |
| `--slice` | ❌ | Window size (base pairs) for promoter scanning (`--pr` only). | `1000` |
| `--stride` | ❌ | Step size for *internal* sliding window within each slice (--pr only). | `200` |
| `--window` | ❌ | Context window size for inference. Default is **Auto-Detect** (Optimal based on Model). | `Auto` |
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
| `--llm-model` | **[New]** Choose Foundation Model: `plantbert`, `dnabert2`, `dnabert1`, or `agront`. | `plantbert` |
| `--epochs` | **[New]** Number of training epochs. | `3` |
| `--kmer` | **[New]** K-mer size for **DNABERT-1** (3, 4, 5, or 6). | `6` |
| `--save-models` | Save valid model checkpoints to `runs/`. Use `--no-save-models` to disable. | `True` |
| `--model-path` | Path to a specific base model (local or HF) if not using presets. | `None` |
| **Data Mining** | | |
| `--organism` | Target scientific name (e.g., "Oryza sativa", "Zea mays"). | "Arabidopsis thaliana" |
| `--keywords` | **[New]** Custom search terms (comma-separated). e.g., "drought,heat". | Standard stress list |
| `--motifs` | **[New]** Custom motif patterns to scan (comma-separated). e.g., "ABRE,MYB". | Standard PLACE list |
| `--limit-genes` | Max number of genes to fetch from NCBI. Set `0` for no limit. | `100` |
| `--task-type` | `binary` (Stress vs Random) or `multiclass` (Stress Type). | `binary` |
| `--max-seq-len` | Maximum length of sequences to download to avoid memory issues. | `20000` |
| `--flank-bp` | Number of base pairs to extract around a detected motif. | `50` |
| **File Paths (Optional)** | | |
| `--gene-list` | Path to a custom .txt file of gene names/accessions. | Auto-generated |
| `--mined-data` | Path to a custom .csv file for training. | Auto-generated |
| `--place-csv` | Path to PLACE database CSV. | Auto-detected |

### 🧬 Dataset Composition Note (New)
The pipeline now creates **Scientifically Robust Datasets**:
1.  **Positive Samples (Label 1)**: Extracted from regions containing known stress motifs (e.g., ABRE, DRE).
2.  **Negative Samples (Label 0)**: Mined from **Genomic Background** (non-motif regions) of the *same genes*. This forces the model to learn the specific stress signature rather than just distinguishing "real DNA" from "random noise".

### 🧬 Available Models

| Model | Size (Params) | Architecture | Best For |
| :--- | :--- | :--- | :--- |
| **PlantBERT** | ~110M | BERT-Base | General purpose, short sequences (<512bp). |
| **DNABERT-2** | ~117M | ALiBi Attention | Long sequences, high efficiency, modern standard. |
| **DNABERT-1** | ~110M | K-mer BERT | Legacy research replication. Requires fixed K-mer. |
| **Agro-NT** | ~1 Billion | Transformer-XL | Large-scale genomic patterns. **High GPU Requirement**. |

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

**3. Train with DNABERT-1 (K-mer based):**
*Legacy support for fixed K-mer tokenization (default 6-mer).*
```bash
python scripts/train.py \
    --step all \
    --organism "Glycine max" \
    --llm-model dnabert1 \
    --kmer 6 \
    --limit-genes 200
```

**4. Dataset Creation Only (No Training):**
```bash
python scripts/train.py \
    --step mine \
    --organism "Triticum aestivum" \
    --limit-genes 1000
```

**5. Custom Traits & Flexible Motifs:**
*Search only for "Heat" related genes and mine only "ABRE" motifs.*
```bash
python scripts/train.py \
    --step all \
    --organism "Solanum lycopersicum" \
    --keywords "heat shock,thermal stress" \
    --motifs "ABRE,HSE" \
    --llm-model dnabert2
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
│   ├── model_utils.py        # ⚙️ Inference Logic
│   └── html_report.py        # 📊 Report Generation Logic
└── software_test/            # 🧪 Test FASTA files
```

---
*Created for Stress Region Prediction Project*
