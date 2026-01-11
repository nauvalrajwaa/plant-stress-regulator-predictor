from stress_predictor import read_fasta, write_output, load_model, get_device, get_args
from stress_predictor import promoter_stress_classification, region_stress_classification

def main():
    args = get_args()

    # Device
    device = get_device(args.force_cpu)
    print(f"Using device: {device}")

    # Read sequences
    sequence = read_fasta(args.input)

    # Process
    if args.pr:
        result = promoter_stress_classification(args.model, args.tokenizer, sequence, device, output_dir=args.output, slice_size=args.slice, stride=args.stride)
    elif args.rg:
        result = region_stress_classification(args.model, args.tokenizer, sequence, device, output_dir=args.output)

    # Write output
    write_output(result, args.output)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()