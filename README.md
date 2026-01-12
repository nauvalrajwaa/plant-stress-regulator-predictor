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

### Basic Usage

**1. Promoter Mode (Long Sequences ~5kb):**
```bash
python stress_predictor/main.py \
    --input "software_test/random_seq_test/random_5kb.fasta" \
    --pr \
    --model "dnabert" \
    --tokenizer "dnabert"
```

**2. Region Mode (Short Sequences ~1kb):**
```bash
python stress_predictor/main.py \
    --input "software_test/random_seq_test/random_2kb.fasta" \
    --rg \
    --model "dnabert" \
    --tokenizer "dnabert"
```

**3. Using a Custom Trained Model (Local Path):**
```bash
python stress_predictor/main.py \
    --input "your_sequence.fasta" \
    --pr \
    --model-path "train_2/plantbert" 
```

### Understanding the Output
Results are saved in `runs/run_YYYYMMDD_HHMMSS_{type}/`.

1.  **`result.json`**: Contains raw probabilities and extracted contiguous stress regions.
2.  **`report.html`**: Interactive HTML visualization of the sequence.
3.  **`heatmap.png`**: (Located in `runs/` folder) Visual representation of stress probability.
    *   **Green**: High Probability of Stress Response.
    *   **Yellow**: Intermediate/Uncertain.
    *   **Red**: Low Probability (Non-Stress).

---

## 🛠 Part 2: Training Pipeline

Use `scripts/train.py` to build your own dataset and train models from scratch using NCBI data.

### Workflow Steps
The pipeline supports `search` -> `mine` -> `check` -> `train` -> `eval`.

### Example Commands

**1. Full Pipeline (Recommended for New Organisms):**
*Searches NCBI, mines sequences, generates logos, and trains the model.*
```bash
python scripts/train.py --step all --organism "Oryza sativa" --limit-genes 500 --email your@email.com
```

**2. Mining Only (Create Dataset):**
*Download sequences for a specific organism without training.*
```bash
python scripts/train.py --step mine --organism "Zea mays" --limit-genes 200 --max-seq-len 5000
```
*Output will be saved in `datasets/dataset_Zea_mays_200_5000_50.csv`*

**3. Multiclass Training:**
*Train a model to distinguish specific stress types (Drought vs Cold vs Salt).*
```bash
python scripts/train.py --step train --task-type multiclass --mined-data "datasets/dataset_MyOrganism.csv"
```

**4. Generate Sequence Logo:**
*Visualize the motifs in your mined dataset.*
```bash
python scripts/train.py --step logo --mined-data "datasets/dataset_MyOrganism.csv" --logo-out "plot/"
```

### Pipeline Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--step` | `all`, `search`, `mine`, `train`, `logo`, `check` | `all` |
| `--organism` | Target scientific name | `Arabidopsis thaliana` |
| `--limit-genes` | Number of genes to process | `100` |
| `--task-type` | `binary` (Stress/No-Stress) or `multiclass` | `binary` |
| `--mined-data` | Input CSV path (Auto-generated if skipped) | `datasets/...` |

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
