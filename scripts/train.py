# scripts/pipeline_manager.py
import argparse
import sys
import os
import re
import time
import pandas as pd
from Bio import Entrez

# Add script directory to path to import ensemble_predictor
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ensemble_predictor import train_multimodel_ml, train_plantbert_from_mined_data, evaluate_models_on_holdout, predict_with_new_models
except ImportError:
    # Try parent directory if running from project root
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    # Using importlib to mimic notebook behavior if needed, but direct import is cleaner
    try:
        from scripts.ensemble_predictor import train_multimodel_ml, train_plantbert_from_mined_data, evaluate_models_on_holdout, predict_with_new_models
    except ImportError:
        print("[ERROR] Could not import ensemble_predictor. Make sure it is in the scripts/ folder.")
        sys.exit(1)

# =============================================================================
# STEP 1: TARGET IDENTIFICATION (NCBI SEARCH)
# =============================================================================
def step1_search_ncbi(email, keywords, organism, output_file="list_gen_stres.txt"):
    """
    Searches NCBI Gene database for Arabidopsis thaliana genes associated with stress keywords.
    """
    print(f"--- STEP 1: SEARCH NCBI GENE DATABASE ---")
    Entrez.email = email
    
    # Query Search
    term_query = f'"{organism}"[Orgn] AND ({ " OR ".join(keywords) })'
    locus_tag_pattern = re.compile(r'AT[1-5MC]G\d{5}', re.IGNORECASE)

    try:
        print(f"[1/3] Sending request to NCBI...")
        search_handle = Entrez.esearch(db="gene", term=term_query, retmax=10000, usehistory="y")
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        count = int(search_results["Count"])
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        
        print(f"      -> Found {count} candidates.")
        
        if count == 0:
            return

        print(f"[2/3] Filtering valid accessions...")
        valid_accessions = set()
        batch_size = 300
        
        for start in range(0, count, batch_size):
            fetch_handle = Entrez.esummary(
                db="gene", 
                retstart=start, 
                retmax=batch_size, 
                webenv=webenv, 
                query_key=query_key
            )
            data = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            for doc in data['DocumentSummarySet']['DocumentSummary']:
                potential_ids = []
                if 'Name' in doc: potential_ids.append(doc['Name'])
                if 'OtherAliases' in doc and doc['OtherAliases']:
                    aliases = doc['OtherAliases'].split(',')
                    potential_ids.extend([a.strip() for a in aliases])

                selected_id = None
                for pid in potential_ids:
                    if locus_tag_pattern.fullmatch(pid):
                        selected_id = pid.upper()
                        break 
                
                if not selected_id and 'Name' in doc:
                    selected_id = doc['Name']
                
                if selected_id:
                    valid_accessions.add(selected_id)
            
            print(f"      -> Progress: {len(valid_accessions)} valid IDs...", end='\r')
            time.sleep(0.5)

        print(f"\n      -> Finished! {len(valid_accessions)} IDs ready.")

        with open(output_file, "w") as f:
            for acc in sorted(valid_accessions):
                f.write(acc + "\n")
        
        print(f"\n[SUCCESS] Saved to: {output_file}")

    except Exception as e:
        print(f"\n[ERROR] {e}")

# =============================================================================
# STEP 2: MINING & MOTIF SCANNING
# =============================================================================
def step2_mine_sequences(email, gene_list_path, task_type, place_csv_path, output_csv="Dataset_Mined.csv", limit_genes=100, organism="Arabidopsis thaliana", max_seq_len=20000, flank_bp=50):
    """
    Downloads sequences and scans for motifs.
    If task_type='multiclass', labels are derived from specific stress keywords found in motifs.
    If task_type='binary', all retrieved sequences are Label=1.
    """
    print(f"\n--- STEP 2: MINING SEQUENCES ---")
    print(f"Config: Organism={organism}, MaxSeqLen={max_seq_len}, FlankBP={flank_bp}, Mode={task_type}")
    Entrez.email = email
    
    # Config
    motif_filter_keywords = ["ABRE", "DRE", "G-BOX", "MYB", "WRKY", "NAC"]
    
    # Load PLACE Motifs
    iu_map = {
        'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'T',
        'R': '[AG]', 'Y': '[CT]', 'M': '[AC]', 'K': '[GT]', 'S': '[GC]', 'W': '[AT]',
        'H': '[ACT]', 'B': '[CGT]', 'V': '[ACG]', 'D': '[AGT]', 'N': '[ACGT]'
    }

    def iupac_to_regex_pattern(seq):
        seq = str(seq).upper().strip()
        seq = "".join([c for c in seq if c in iu_map]) 
        res = ""
        for c in seq: res += iu_map.get(c, c)
        return res

    print(f"[1/3] Loading PLACE Motif Library from {place_csv_path}...")
    motif_library = []
    if os.path.exists(place_csv_path):
        df_place = pd.read_csv(place_csv_path)
        col_seq = 'Sequence' if 'Sequence' in df_place.columns else df_place.columns[1]
        col_id = 'ID' if 'ID' in df_place.columns else df_place.columns[0]

        for idx, row in df_place.iterrows():
            m_id = str(row[col_id]).upper()
            m_seq = row[col_seq]
            
            is_match = False
            matched_category = None
            
            for k in motif_filter_keywords:
                if k.upper() in m_id:
                    is_match = True
                    matched_category = k # Capture the category (e.g., ABRE)
                    break
            
            if is_match and len(str(m_seq)) >= 3:
                motif_library.append({
                    'id': row[col_id],
                    'original': m_seq,
                    'regex': re.compile(iupac_to_regex_pattern(m_seq)),
                    'category': matched_category
                })
        print(f"      -> Selected {len(motif_library)} motifs.")
    else:
        print(f"[ERROR] PLACE CSV not found: {place_csv_path}")
        return

    # Fetch Function
    def fetch_sequence_strict(locus_tag):
        try:
            query = f"{locus_tag}[Gene Name] AND {organism}[Orgn] AND refseq[filter] AND biomol_mrna[PROP]"
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            
            if not record['IdList']: return None, "No mRNA found"
            seq_id = record['IdList'][0]
            
            with Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text") as handle:
                fasta_data = handle.read()
            
            lines = fasta_data.strip().split('\n')
            if not lines: return None, "Empty FASTA"
            full_seq = "".join(lines[1:]).upper()
            
            if len(full_seq) > max_seq_len: return None, f"OVERSIZE ({len(full_seq)} bp)"
            return full_seq, "Success"
        except Exception as e:
            return None, str(e)

    # Main Loop
    print(f"[2/3] Loading Target Genes from {gene_list_path}")
    if os.path.exists(gene_list_path):
        with open(gene_list_path, 'r') as f:
            all_genes = [line.strip() for line in f if line.strip()]
        target_genes = all_genes[:limit_genes] if limit_genes else all_genes
    else:
        print(f"[ERROR] Gene list not found.")
        return

    print(f"[3/3] Mining {len(target_genes)} genes...")
    results = []
    
    for i, gid in enumerate(target_genes):
        print(f"[{i+1}/{len(target_genes)}] {gid}: ", end="")
        seq, status = fetch_sequence_strict(gid)
        
        if seq:
            print(f"OK ({len(seq)}bp). Scan...", end=" ")
            hits = 0
            for m in motif_library:
                for match in m['regex'].finditer(seq):
                    start, end = match.start(), match.end()
                    if start >= flank_bp and (end + flank_bp) <= len(seq):
                        
                        # Apply Multiclass Logic
                        label = 1
                        if task_type == 'multiclass':
                            label = m['category'] # Use the keyword ("ABRE", "MYB") as label

                        results.append({
                            'Gene_Locus': gid,
                            'Motif_ID': m['id'],
                            'Motif_Pattern': m['original'],
                            'Extracted_Sequence': seq[start-flank_bp : end+flank_bp],
                            'Label': label
                        })
                        hits += 1
            print(f"Found {hits}.")
        else:
            print(f"SKIP -> {status}")
        
        # Save occasionally
        if len(results) > 0 and i % 10 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)

    if results:
        final_df = pd.DataFrame(results).drop_duplicates(subset=['Extracted_Sequence'])
        final_df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Mined data saved: {output_csv} ({len(final_df)} records)")
    else:
        print("\n[WARN] No data found.")

# =============================================================================
# STEP 3: TRAINING
# =============================================================================
def step3_train(mined_csv, task_type):
    print(f"\n--- STEP 3: TRAINING MODELS ({task_type}) ---")
    
    # 1. ML Benchmark
    best_ml_model, vectorizer = train_multimodel_ml(mined_csv, task_type=task_type)
    
    # 2. PlantBERT
    plantbert_model, bert_tokenizer = train_plantbert_from_mined_data(mined_csv, task_type=task_type)
    
    return best_ml_model, vectorizer, plantbert_model, bert_tokenizer

# =============================================================================
# MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stress Region Predictor Training Pipeline",
        epilog="""
Examples:
  1. Full pipeline (Search -> Mine -> Train):
     python scripts/train.py --step all --email user@example.com

  2. Mining only (with custom organism and size):
     python scripts/train.py --step mine --organism "Oryza sativa" --limit-genes 500

  3. Multiclass Training:
     python scripts/train.py --step train --task-type multiclass --mined-data "My_Rice_Data.csv"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # General args
    parser.add_argument("--step", type=str, choices=["all", "search", "mine", "train", "eval"], default="all", help="Pipeline step to run")
    parser.add_argument("--email", type=str, default="user@example.com", help="NCBI Email")
    parser.add_argument("--task-type", type=str, choices=["binary", "multiclass"], default="binary", help="Classification type")
    
    # New Configurable Arguments
    parser.add_argument("--organism", type=str, default="Arabidopsis thaliana", help="Target Organism (e.g. 'Oryza sativa')")
    parser.add_argument("--max-seq-len", type=int, default=20000, help="Maximum sequence length to download")
    parser.add_argument("--flank-bp", type=int, default=50, help="Flanking base pairs around motif")
    
    # Paths
    parser.add_argument("--gene-list", type=str, default=None, help="Path to gene list file (Auto-generated if None)")
    parser.add_argument("--place-csv", type=str, default="/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/PLACE_Parsed_Complete_V2.csv", help="Path to PLACE motif CSV")
    parser.add_argument("--mined-data", type=str, default=None, help="Path to output mined CSV (Auto-generated if None)")
    parser.add_argument("--limit-genes", type=int, default=100, help="Limit number of genes to mine (0 = No limit)")
    
    args = parser.parse_args()

    # Dynamic Path Generation
    # Create datasets folder if not exists
    datasets_dir = os.path.join(current_dir, "..", "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # Sanitize organism name for filename (remove spaces)
    org_clean = args.organism.replace(" ", "_")
    run_id = f"{org_clean}_{args.limit_genes}_{args.max_seq_len}_{args.flank_bp}"
    
    # Set default paths if not provided
    if args.gene_list is None:
        args.gene_list = os.path.join(datasets_dir, f"list_gen_{run_id}.txt")
    
    if args.mined_data is None:
        args.mined_data = os.path.join(datasets_dir, f"dataset_{run_id}.csv")
        
    print(f"--- RUN CONFIGURATION ---")
    print(f"Run ID: {run_id}")
    print(f"Output Directory: {datasets_dir}")
    print(f"Gene List: {args.gene_list}")
    print(f"Mined Sequence Data: {args.mined_data}")
    print(f"-------------------------")
    
    # Define Keywords for Search
    stress_keywords = [
        "drought", "salt stress", "cold stress", "heat shock", 
        "abscisic acid", "water deprivation", "oxidative stress", "salinity"
    ]

    # Execute Flow
    if args.step in ["all", "search"]:
        step1_search_ncbi(args.email, stress_keywords, args.organism, args.gene_list)
        
    if args.step in ["all", "mine"]:
        limit = args.limit_genes if args.limit_genes > 0 else None
        step2_mine_sequences(args.email, args.gene_list, args.task_type, args.place_csv, args.mined_data, limit, args.organism, args.max_seq_len, args.flank_bp)
        
    if args.step in ["all", "train"]:
        ml_model, vect, bert_model, bert_tok = step3_train(args.mined_data, args.task_type)
        
        # Immediate Evaluation if 'train' is independent step or part of 'all'
        if args.step == "train" or args.step == "all":
            print("\n--- STEP 4: EVALUATION ---")
            evaluate_models_on_holdout(args.mined_data, ml_model, vect, bert_model, bert_tok)
            predict_with_new_models(args.mined_data, ml_model, vect, bert_model, bert_tok)

    if args.step == "eval":
        # Note: Eval usually requires models in memory. 
        # Loading saved models logic needs to be added if running standalone 'eval'
        print("Standalone evaluation requires loading saved models. Please run 'train' step which includes evaluation.")

if __name__ == "__main__":
    main()
