import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

try:
    import logomaker
except ImportError:
    logomaker = None
    print("[WARN] logomaker not installed. Logo generation will be skipped.")

# IUPAC codes for checking
IUPAC_CHARS = set(list("ACGTURYSWKMBDHVNacgturyswkmbdhvn"))

def check_sequence(seq: str):
    if pd.isna(seq):
        return False, 'missing'
    s = str(seq).upper().strip()
    if s == '':
        return False, 'empty'
    bad = [c for c in s if c not in IUPAC_CHARS]
    if bad:
        return False, f'illegal_chars:{"".join(sorted(set(bad)))}'
    return True, 'ok'

def analyze_stats(df, col_seq='Extracted_Sequence', col_id='Label'):
    print("\n--- STATISTICAL ANALYSIS ---")
    
    # 1. Length Analysis
    if col_seq in df.columns:
        df['Length'] = df[col_seq].astype(str).str.strip().apply(len)
        print("Sequence Length Statistics:")
        print(df['Length'].describe())

        print("\nTop 5 shortest sequences:")
        print(df.nsmallest(5, 'Length')[[col_id, col_seq, 'Length']])

        print("\nTop 5 longest sequences:")
        print(df.nlargest(5, 'Length')[[col_id, col_seq, 'Length']])
    else:
        print(f"[WARN] Sequence column '{col_seq}' not found for stats.")

    # 2. Class Analysis
    if col_id in df.columns:
        num_classes = df[col_id].nunique()
        print(f"\nNumber of unique Classes ({col_id}): {num_classes}")

        print("\nTop 20 most frequent classes:")
        print(df[col_id].value_counts().head(20))

        # Check for singletons
        class_counts = df[col_id].value_counts()
        singletons = class_counts[class_counts == 1]
        print(f"\nNumber of classes with only 1 sample: {len(singletons)}")
        print(f"Number of classes with < 5 samples: {len(class_counts[class_counts < 5])}")
    else:
        print(f"[WARN] Class/ID column '{col_id}' not found for stats.")

def check_and_save(df, out_path, col_seq='Extracted_Sequence'):
    print(f"\n--- SEQUENCE VALIDATION ---")
    if col_seq not in df.columns:
        print(f"[ERROR] Column '{col_seq}' not found.")
        return

    results = []
    for idx, row in df.iterrows():
        seq = row.get(col_seq, '')
        valid, reason = check_sequence(seq)
        row_dict = row.to_dict()
        row_dict['Valid'] = valid
        row_dict['Reason'] = reason
        results.append(row_dict)

    df_out = pd.DataFrame(results)
    if out_path:
        df_out.to_csv(out_path, index=False)
        print(f"Saved validated data to: {out_path}")
        
        invalid_count = len(df_out[df_out['Valid'] == False])
        if invalid_count > 0:
            print(f"[WARN] Found {invalid_count} invalid sequences.")
        else:
            print("[SUCCESS] All sequences are valid IUPAC strings.")

def create_logo(df, out_dir, name, col_seq='Extracted_Sequence', clean_noise=True, seq_type='dna'):
    if logomaker is None:
        return

    print(f"\n--- GENERATING LOGO ({name}) ---")

    if col_seq not in df.columns:
        print(f"[ERROR] Column '{col_seq}' not found.")
        return

    # Get sequences
    sequences = df[col_seq].dropna().astype(str).tolist()
    if not sequences:
        print("[ERROR] No data.")
        return

    # Filter length to be uniform (most common length)
    lengths = [len(s) for s in sequences]
    if not lengths:
        return
        
    most_common_len = max(set(lengths), key=lengths.count)
    valid_sequences = [s for s in sequences if len(s) == most_common_len]
    print(f"Processing {len(valid_sequences)} sequences (Length: {most_common_len})...")
    
    if len(valid_sequences) < 5:
        print("[WARN] Too few sequences to generate logo.")
        return

    # Matrix
    try:
        if seq_type.lower() in ('dna', 'nucleotide'):
            chars_to_ignore = '.-X*'
            color_scheme = 'classic'
        else:
            chars_to_ignore = '.-X*'
            color_scheme = 'skylign_protein'

        ww_counts = logomaker.alignment_to_matrix(sequences=valid_sequences, to_type='information', characters_to_ignore=chars_to_ignore)
        
        if clean_noise:
            ww_counts[ww_counts < 0.10] = 0.0
            
    except Exception as e:
        print(f"[ERROR] Matrix creation failed: {e}")
        return

    # Plot
    try:
        fig_width = max(8, len(valid_sequences[0]) * 0.4) 
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        logo = logomaker.Logo(ww_counts,
                              color_scheme=color_scheme,
                              vpad=.05,
                              width=.9,
                              ax=ax)

        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        ax.set_ylabel("Bits", fontsize=14)
        ax.set_xlabel("Position", fontsize=14)
        
        if seq_type.lower() in ('dna', 'nucleotide'):
             ax.set_ylim([0, 2.1])
        else:
             ax.set_ylim([0, 4.5])
             
        step = max(1, len(ww_counts) // 20)
        ax.set_xticks(range(0, len(ww_counts), step))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        output_path = os.path.join(out_dir, f"{name}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig) 
        print(f"[SUCCESS] Logo saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and Visualize Mined Data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Mode selection
    parser.add_argument('--mode', choices=['all', 'stats', 'check', 'logo'], default='all', help='Analysis mode')

    # Input/Output
    parser.add_argument('--input', type=str, default=None, help='Input CSV file path. If None, attemps to construct from params.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results (default: same as input or datasets/)')

    # Column mappings
    parser.add_argument('--col-seq', type=str, default='Extracted_Sequence', help='Column name for sequence')
    parser.add_argument('--col-id', type=str, default='Label', help='Column name for ID/Class')

    # Params for auto-path construction (same as train.py)
    parser.add_argument("--organism", type=str, default="Arabidopsis thaliana", help="Target Organism")
    parser.add_argument("--limit-genes", type=int, default=100, help="Limit genes")
    parser.add_argument("--max-seq-len", type=int, default=20000, help="Max seq len")
    parser.add_argument("--flank-bp", type=int, default=50, help="Flank bp")
    
    args = parser.parse_args()

    # Determine Input File
    if args.input:
        input_path = args.input
    else:
        # Try to match train.py logic
        current_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(current_dir, "..", "datasets")
        org_clean = args.organism.replace(" ", "_")
        run_id = f"{org_clean}_{args.limit_genes}_{args.max_seq_len}_{args.flank_bp}"
        input_path = os.path.join(datasets_dir, f"dataset_{run_id}.csv")
        print(f"Auto-detected input path: {input_path}")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    # Determine Output Dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        # Save in 'analysis' folder inside the input file's directory
        parent = os.path.dirname(input_path)
        out_dir = os.path.join(parent, "analysis")
    
    os.makedirs(out_dir, exist_ok=True)

    # Load Data
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} records from {input_path}")
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(1)
        
    # Column fallback checks (compatibility with raw PLACE or mined data)
    if args.col_seq not in df.columns:
        if 'Sequence' in df.columns: args.col_seq = 'Sequence'
        elif 'sequence' in df.columns: args.col_seq = 'sequence'
    
    if args.col_id not in df.columns:
        if 'ID' in df.columns: args.col_id = 'ID'
        elif 'Gene_Locus' in df.columns: args.col_id = 'Gene_Locus'

    # Execute Modes
    if args.mode in ['all', 'stats']:
        analyze_stats(df, args.col_seq, args.col_id)

    if args.mode in ['all', 'check']:
        out_check = os.path.join(out_dir, os.path.basename(input_path).replace('.csv', '_checked.csv'))
        check_and_save(df, out_check, args.col_seq)

    if args.mode in ['all', 'logo']:
        # Generate generic logo for full dataset
        create_logo(df, out_dir, f"logo_full", args.col_seq)
        
        # Optional: Generate logo per class if enough data
        if args.col_id in df.columns:
            print("\n--- GENERATING PER-CLASS LOGOS ---")
            top_classes = df[args.col_id].value_counts().head(5).index.tolist()
            for cls in top_classes:
                df_cls = df[df[args.col_id] == cls]
                if len(df_cls) > 10:
                    safe_cls = str(cls).replace("/", "_").replace(" ", "_")
                    create_logo(df_cls, out_dir, f"logo_class_{safe_cls}", args.col_seq)

if __name__ == "__main__":
    main()
