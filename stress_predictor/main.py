import sys
import os
from datetime import datetime

# Add project root to path so we can import 'stress_predictor' packages
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from stress_predictor import read_fasta, write_output, load_model, get_device, get_args
from stress_predictor import promoter_stress_classification, region_stress_classification

from stress_predictor.io_utils import generate_html_report # Explicit import needed if __init__.py not updated yet

def main():
    args = get_args()

    # Determine Output Directory
    if args.output is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Define run type for clearer folder naming
        run_type = "promoter" if args.pr else "region"
        args.output = os.path.join("runs", f"run_{run_id}_{run_type}")
    
    # Create output directory immediately
    os.makedirs(args.output, exist_ok=True)
    print(f"Run ID: {os.path.basename(args.output)}")
    print(f"Output directory: {args.output}")

    # Device
    device = get_device(args.force_cpu)
    print(f"Using device: {device}")

    # Read sequences
    sequence = read_fasta(args.input)

    # Process
    mode = "rg"
    if args.pr:
        mode = "pr"
        result = promoter_stress_classification(
            args.model, args.tokenizer, sequence, device, 
            output_dir=args.output, slice_size=args.slice, stride=args.stride,
            model_path=args.model_path
        )
    elif args.rg:
        result = region_stress_classification(
            args.model, args.tokenizer, sequence, device, 
            output_dir=args.output,
            model_path=args.model_path
        )

    # Write output
    write_output(result, args.output)
    generate_html_report(result, args.output, mode=mode)
    
    print(f"Predictions and HTML report saved to {args.output}")

if __name__ == "__main__":
    main()