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

from stress_predictor.html_report import generate_html_report # Explicit import needed if __init__.py not updated yet

class DualLogger:
    """Redirects stdout to both terminal and a file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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
    
    # Setup logging to file
    log_file = os.path.join(args.output, "analysis_log.txt")
    sys.stdout = DualLogger(log_file)

    print("\n" + "="*60)
    print(f"🌱 PLANT STRESS PREDICTOR | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Processing Run ID: {os.path.basename(args.output)}")
    print(f"Output Directory : {args.output}")
    print("-" * 60)
    
    # Device
    device = get_device(args.force_cpu)
    print(f"Compute Device   : {str(device).upper()}")
    if "cuda" in str(device).lower():
         print(f"GPU Name         : {torch.cuda.get_device_name(0)}")
    print("-" * 60 + "\n")

    # Read sequences
    sequence = read_fasta(args.input)

    # Process
    mode = "rg"
    if args.pr:
        mode = "pr"
        result = promoter_stress_classification(
            args.model, args.tokenizer, sequence, device, 
            output_dir=args.output, slice_size=args.slice, stride=args.stride,
            window_size=args.window,
            model_path=args.model_path
        )
    elif args.rg:
        result = region_stress_classification(
            args.model, args.tokenizer, sequence, device, 
            output_dir=args.output,
            window_size=args.window,
            model_path=args.model_path
        )

    # Write output
    write_output(result, args.output)
    generate_html_report(result, args.output, mode=mode)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print(f"📂 Results saved to : {args.output}")
    print(f"� Run Log          : {log_file}")
    print(f"�📊 HTML Report      : {os.path.join(args.output, f'stress_report_{mode}.html')}") 
    print("="*60 + "\n")

if __name__ == "__main__":
    main()