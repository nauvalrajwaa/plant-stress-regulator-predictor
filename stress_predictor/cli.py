import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Stress Predictor CLI with Hugging Face model")
    parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--model", type=str, help="Hugging Face model name, choose between dnabert or mistral-athaliana")
    parser.add_argument("--tokenizer", type=str, help="Hugging Face tokenizer name, choose between dnabert or mistral-athaliana")
    parser.add_argument("--model-path", type=str, help="Path to local model directory (overrides Hugging Face model name)")
    parser.add_argument("--force-cpu", action="store_true", help="Force using CPU even if GPU is available")

    # mutually exclusive mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--pr", action="store_true", help="Enable promoter detection mode")
    mode_group.add_argument("--rg", action="store_true", help="Enable region-only mode")

    # optional args (need validation)
    parser.add_argument("--slice", type=int, help="Chunk size (only for --pr). Default: 1000")
    parser.add_argument("--stride", type=int, help=("Stride size (only for --pr)"
                                                    "Valid values depend on --slice-size: "
                                                    "1000 → [100, 200, 500]; "
                                                    "2000 → [100, 200, 400, 500]. "
                                                    "Default: 200."))
    parser.add_argument("--output", type=str, help=("Output folder. "
                                                    "Default: runs/run_<timestamp>_<type>"))

    args = parser.parse_args()
    
    # Validation: Either model+tokenizer OR model-path must be provided
    if args.model_path:
        # Local model logic, model/tokenizer args ignored if present
        pass
    else:
        if not args.model or not args.tokenizer:
            parser.error("Arguments --model and --tokenizer are required unless --model-path is provided.")

    if args.pr:
        if args.slice is None:
            args.slice = 1000
        if args.stride is None:
            args.stride = 200
        # Output default handling moved to main.py to support 'runs' folder structure
    elif args.rg:
        # Output default handling moved to main.py to support 'runs' folder structure
        if args.slice is not None or args.stride is not None:
            parser.error("--slice and --stride are only valid with --pr")

    return args
