# Stress Predictor

A high-performance command-line interface (CLI) tool designed to predict stress-responsive regions in DNA sequences. By leveraging state-of-the-art Transformer models fine-tuned on plant genomic data, it provides researchers with specific insights into genetic stress responses.

## 🌟 Key Features

- **Advanced Transformer Models**: Supports both **DNABERT-2** and **Mistral DNA Athaliana** specialized architectures.
- **Optimized Inference Engine**: Refactored prediction logic for high efficiency—up to **100x faster** than standard implementations by minimizing model loading overhead.
- **Dual Prediction Modes**: 
  - **Region Mode**: Targeted analysis of small genomic sequences (1-2kb).
  - **Promoter Mode**: Large-scale analysis of promoter regions (5-10kb) using intelligent slicing.
- **Robust Error Handling**: Built-in CUDA Out-of-Memory (OOM) recovery with automatic CPU fallback.
- **Visual Analytics**: Automatically generates heatmaps and sequence-level visualizations of stress probability.

## 🏗️ Project Structure
```text
stress-predictor/
├── pyproject.toml           # Modular Project configuration & dependencies
├── stress_predictor/
│   ├── main.py              # Application entry point
│   ├── cli.py               # Argument parsing logic
│   ├── io_utils.py          # Validated FASTA handling
│   ├── model_utils.py       # core inference engine & performance fixes
│   └── __init__.py          # Module exports
└── software_test/           # Validation datasets and baseline results
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/venusangela/stress-predictor.git
cd stress-predictor

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### 2. Basic Usage

#### **Region Classification** (1kb - 2kb sequences)
```bash
stress-predictor --rg --input sequences.fasta --model dnabert --tokenizer dnabert --output results_rg
```

#### **Promoter Classification** (5kb - 10kb sequences)
```bash
stress-predictor --pr --input promoter.fasta --model mistral-athaliana --tokenizer mistral-athaliana --output results_pr
```

## 📊 Output Formats

The tool generates two types of outputs in your specified directory:

1.  **`result.json`**: A detailed report containing:
    *   Window-by-window prediction labels.
    *   Confidence scores for every genomic segment.
    *   Aggregated final stress score for the entire sequence.
2.  **Visualizations (`.png`)**: 
    *   Heatmaps showing predicted stress regions across the sequence length.
    *   Color-coded segments (Green for Stress, Red for Non-Stress).

## 🧠 Supported Models

| Model Name | Architecture | Target Plant | Source |
| :--- | :--- | :--- | :--- |
| `dnabert` | BERT-based | *Nicotiana tabaccum* | [HuggingFace: igemugm](https://huggingface.co/igemugm) |
| `mistral-athaliana` | Mistral-based | *Arabidopsis thaliana* | [HuggingFace: igemugm](https://huggingface.co/igemugm) |

## 🛠️ Configuration & Parameters

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to FASTA file (Single sequence) | **Required** |
| `--model` | Model name (`dnabert` or `mistral-athaliana`) | **Required** |
| `--force-cpu` | Force inference on CPU even if GPU is present | `False` |
| `--slice` | Window size for promoter slicing | `1000` |
| `--stride` | Stride/overlap between slices | `200` |

## ❗ Troubleshooting

*   **CUDA Out of Memory**: The tool will automatically attempt to clear cache or switch to CPU. If it persists, try increasing the `--stride`.
*   **Command Not Found**: Ensure you have activated your virtual environment and ran `pip install -e .`.
*   **Sequence Length Error**: Ensure your FASTA file contains only **one** sequence and matches the required length (1-2kb for RG, 5-10kb for PR).

---
*Created by iGEM UGM (igemugm@gmail.com)*
