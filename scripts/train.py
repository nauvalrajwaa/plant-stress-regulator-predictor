# scripts/pipeline_manager.py
import argparse
import sys
import os
import re
import time
import pandas as pd
from Bio import Entrez
from urllib.error import HTTPError  # FIX: Import library standar untuk handling error jaringan

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
        # Define dummy functions if ML modules are missing to allow mining to proceed
        def train_multimodel_ml(*args, **kwargs): return None, None
        def train_plantbert_from_mined_data(*args, **kwargs): return None, None
        def evaluate_models_on_holdout(*args, **kwargs): pass
        def predict_with_new_models(*args, **kwargs): pass
        print("[WARN] Could not import ensemble_predictor. ML Training steps will be skipped.")

# =============================================================================
# STEP 1: TARGET IDENTIFICATION (NCBI SEARCH)
# =============================================================================
def step1_search_ncbi(email, keywords, organism, output_file="list_gen_stres.txt"):
    """
    Searches NCBI Gene database, with fallback to Nucleotide database if Gene DB fails.
    """
    print(f"--- STEP 1: SEARCH NCBI DATABASE ---")
    Entrez.email = email
    Entrez.tool = "StressRegionPredictor"
    
    # Query Search
    quoted_keywords = [f'"{k}"' for k in keywords]
    base_query = f'"{organism}"[Orgn] AND ({ " OR ".join(quoted_keywords) })'
    
    candidates = set()
    search_success = False

    # --- STRATEGY A: Try GENE Database ---
    try:
        print(f"[1/3] Attempting search in 'gene' database...")
        search_handle = Entrez.esearch(db="gene", term=base_query, retmax=5000, usehistory="y")
        try:
            search_results = Entrez.read(search_handle, validate=False)
        except TypeError:
            search_results = Entrez.read(search_handle)
        search_handle.close()
        
        count = int(search_results["Count"])
        
        if count > 0:
            print(f"      -> Found {count} candidates in Gene DB.")
            webenv = search_results["WebEnv"]
            query_key = search_results["QueryKey"]
            
            batch_size = 300
            for start in range(0, count, batch_size):
                fetch_handle = Entrez.esummary(db="gene", retstart=start, retmax=batch_size, webenv=webenv, query_key=query_key)
                try:
                    data = Entrez.read(fetch_handle, validate=False)
                except TypeError:
                    data = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                doc_set = data.get('DocumentSummarySet', {}).get('DocumentSummary', [])
                for doc in doc_set:
                    if 'Name' in doc: candidates.add(doc['Name'])
                    elif 'OtherAliases' in doc and doc['OtherAliases']:
                        candidates.add(doc['OtherAliases'].split(',')[0].strip())
                
                print(f"      -> Progress: {len(candidates)} IDs...", end='\r')
            search_success = True
        else:
            print("      -> No results in Gene DB.")

    except Exception as e:
        print(f"      [WARN] Gene DB search failed: {e}")

    # --- STRATEGY B: Fallback to NUCLEOTIDE Database ---
    if not search_success or len(candidates) == 0:
        print(f"\n      -> Switching to 'nucleotide' database fallback...")
        try:
            nucleotide_query = base_query + " AND biomol_mrna[PROP] AND refseq[filter]"
            search_handle = Entrez.esearch(db="nucleotide", term=nucleotide_query, retmax=5000, usehistory="y")
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            count = int(search_results["Count"])
            print(f"      -> Found {count} mRNA sequences.")
            
            if count > 0:
                webenv = search_results["WebEnv"]
                query_key = search_results["QueryKey"]
                batch_size = 300
                
                for start in range(0, count, batch_size):
                    fetch_handle = Entrez.esummary(db="nucleotide", retstart=start, retmax=batch_size, webenv=webenv, query_key=query_key)
                    data = Entrez.read(fetch_handle, validate=False)
                    fetch_handle.close()
                    
                    for doc in data:
                        # Prioritize Caption (Accession)
                        acc = doc.get('Caption') or doc.get('AccessionVersion')
                        if acc: candidates.add(acc)
                    print(f"      -> Progress: {len(candidates)} IDs...", end='\r')

        except Exception as e:
            print(f"\n[ERROR] Nucleotide search also failed: {e}")

    print(f"\n      -> Finished! {len(candidates)} IDs ready.")

    if candidates:
        with open(output_file, "w") as f:
            for acc in sorted(candidates):
                f.write(acc + "\n")
        print(f"\n[SUCCESS] Saved to: {output_file}")
    else:
        print(f"\n[ERROR] No candidates found. Check keywords/organism.")

# =============================================================================
# STEP 2: MINING & MOTIF SCANNING
# =============================================================================
def step2_mine_sequences(email, gene_list_path, task_type, place_csv_path, output_csv="Dataset_Mined.csv", limit_genes=100, organism="Arabidopsis thaliana", max_seq_len=20000, flank_bp=50):
    """
    Downloads sequences and scans for motifs. Supports both Gene Names and Accession IDs.
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
                    matched_category = k
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

    # Fetch Function (SMART)
    def fetch_sequence_smart(target_id):
        try:
            # Check if input looks like an Accession (e.g., NM_1234, XM_5678)
            is_accession = re.match(r'^[A-Z]{2}_?\d+(\.\d+)?$', target_id, re.IGNORECASE)
            seq_id = None
            
            if is_accession:
                seq_id = target_id
            else:
                # Treat as Gene Name
                query = f"{target_id}[Gene Name] AND {organism}[Orgn] AND refseq[filter] AND biomol_mrna[PROP]"
                handle = Entrez.esearch(db="nucleotide", term=query, retmax=1)
                record = Entrez.read(handle)
                handle.close()
                if record['IdList']:
                    seq_id = record['IdList'][0]

            if not seq_id:
                return None, "Not found"
            
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
        sys.stdout.flush()
        
        # FIX: Use smart fetch
        seq, status = fetch_sequence_smart(gid)
        
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
                            label = m['category']

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
def step3_train(mined_csv, task_type, organism="Unknown", models_dir=None, save_models=True, base_model_path=None):
    print(f"\n--- STEP 3: TRAINING MODELS ({task_type}) ---")
    
    try:
        # 1. ML Benchmark
        # Pass organism and output dir to allow naming: {Model}_{Acc}_{Organism}.pkl
        best_ml_model, vectorizer = train_multimodel_ml(
            mined_csv, 
            output_dir=models_dir, 
            organism=organism, 
            task_type=task_type,
            save_model=save_models
        )
        
        # 2. PlantBERT
        # Pass organism and output dir
        plantbert_model, bert_tokenizer = train_plantbert_from_mined_data(
            mined_csv, 
            output_dir=models_dir, 
            organism=organism,
            task_type=task_type,
            save_model=save_models,
            base_model_path=base_model_path
        )
        
        return best_ml_model, vectorizer, plantbert_model, bert_tokenizer
    except Exception as e:
        print(f"[ERROR] Training crashed: {e}")
        # traceback would be good here, but keeping it simple
        import traceback
        traceback.print_exc()
        return None, None, None, None

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument("--place-csv", type=str, default=None, help="Path to PLACE motif CSV (Auto-detected if None)")
    parser.add_argument("--mined-data", type=str, default=None, help="Path to output mined CSV (Auto-generated if None)")
    parser.add_argument("--limit-genes", type=int, default=100, help="Limit number of genes to mine (0 = No limit)")
    parser.add_argument("--logo-out", type=str, default=None, help="Output directory for Logo plots")
    
    # Save Model Flag
    parser.add_argument("--save-models", action="store_true", default=True, help="Save trained models (default: True)")
    parser.add_argument("--no-save-models", action="store_false", dest="save_models", help="Disable model saving")
    
    # Model & LLM Flags
    # --model-path takes precedence if provided. --llm-model sets --model-path to presets.
    parser.add_argument("--model-path", type=str, default=None, help="Path to base model (PlantBERT or DNABERT, local or HF ID)")
    parser.add_argument("--llm-model", type=str, default="plantbert", choices=["plantbert", "dnabert2", "custom"], 
                        help="Select LLM preset: 'plantbert' (nigelhartm/PlantBERT) or 'dnabert2' (zhihan1996/DNABERT-2-117M)")

    args = parser.parse_args()

    # Dynamic Path Generation
    # Define project root
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # Sanitize organism name for filename (remove spaces)
    org_clean = args.organism.replace(" ", "_").replace("(", "").replace(")", "")
    
    # Create Unique RUN ID based on timestamp and config
    # Format: run_{YYYYMMDD_HHMMSS}_{Organism}_{ModelType}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_type_tag = "dnabert" if (args.llm_model == "dnabert2" or (args.model_path and "dnabert" in args.model_path.lower())) else "plantbert"
    run_id = f"run_{timestamp}_{org_clean}_{model_type_tag}"
    
    # Define Output Directory for THIS specific run
    runs_dir = os.path.join(project_root, "runs")
    current_run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(current_run_dir, exist_ok=True)
    
    # Define Models Directory (Global or Run Specific?)
    # User requested separate runs folder logic.
    # We will save models INSIDE this run folder to keep everything together.
    models_dir = os.path.join(current_run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)       
    
    # Define Datasets Directory (Global or Run Specific?)
    # Keeping raw datasets global is usually better to avoid duplication, 
    # BUT finding them inside the run folder is clearer for reproducibility.
    # Let's place the specific dataset for this run inside the run folder.
    datasets_dir = os.path.join(current_run_dir, "data")
    os.makedirs(datasets_dir, exist_ok=True)

    # Resolve Model Path based on LLM Selection
    # If explicit --model-path is given, it overrides the preset.
    if args.model_path is None:
        if args.llm_model.lower() == "dnabert2":
             args.model_path = "zhihan1996/DNABERT-2-117M"
        elif args.llm_model.lower() == "plantbert":
             # Default fallback is local PlantBERT if present, else HF ID
             # Checking if local PlantBERT exists to maintain backward compatibility
             local_plantbert = os.path.join(project_root, "PlantBERT")
             if os.path.exists(local_plantbert):
                 args.model_path = local_plantbert
                 print(f"[INFO] Using local PlantBERT found at: {local_plantbert}")
             else:
                 args.model_path = "nigelhartm/PlantBERT"
        else:
             # Default generic fallback
             args.model_path = os.path.join(project_root, "PlantBERT")
    
    # Resolve PLACE CSV
    if args.place_csv is None:
        possible_paths = [
             os.path.join(project_root, "train_1", "data", "PLACE_Parsed_Complete_V2.csv"),
             os.path.join(project_root, "train", "data", "PLACE_Parsed_Complete_V2.csv"),
             os.path.join(project_root, "data", "PLACE_Parsed_Complete_V2.csv")
        ]
        for p in possible_paths:
             if os.path.exists(p):
                 args.place_csv = p
                 break
        if args.place_csv is None:
             print("[WARN] Could not auto-detect PLACE_Parsed_Complete_V2.csv. Please provide --place-csv.")

    # Create datasets folder if not exists
    # datasets_dir handled above in RUN ID logic

    # run_id variable already created above
    # simple_id for file naming
    file_id = f"{org_clean}_{args.limit_genes}_{args.max_seq_len}"
    
    # Set default paths if not provided
    if args.gene_list is None:
        args.gene_list = os.path.join(datasets_dir, f"list_gen_{file_id}.txt")
    
    if args.mined_data is None:
        args.mined_data = os.path.join(datasets_dir, f"dataset_{file_id}.csv")
        
    print(f"--- RUN CONFIGURATION ---")
    print(f"Run ID: {run_id}")
    print(f"Run Directory: {current_run_dir}")
    print(f"Gene List: {args.gene_list}")
    print(f"Mined Sequence Data: {args.mined_data}")
    print(f"Models Directory: {models_dir if args.save_models else 'Not saving'}")
    print(f"Base Model Path: {args.model_path}")
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
        # Update call to accept new arguments (organism, models_dir, save_models)
        ml_model, vect, bert_model, bert_tok = step3_train(
            args.mined_data, 
            args.task_type,
            organism=args.organism,
            models_dir=models_dir, # Now pointing to runs/run_ID/models
            save_models=args.save_models,
            base_model_path=args.model_path
        )
        
        if ml_model is None or bert_model is None:
            print("\n[ERROR] Training returned None. Please check if mined data exists and is not empty.")
        
        # Immediate Evaluation if 'train' is independent step or part of 'all'
        if (args.step == "train" or args.step == "all") and ml_model is not None:
            print("\n--- STEP 4: EVALUATION ---")
            
            # Predict and Save Results to RUN Directory
            results_df = predict_with_new_models(args.mined_data, ml_model, vect, bert_model, bert_tok)
            
            if results_df is not None:
                 res_path = os.path.join(current_run_dir, "predictions.csv")
                 results_df.to_csv(res_path, index=False)
                 print(f"      -> Final Predictions saved to: {res_path}")

    if args.step == "eval":
        # Note: Eval usually requires models in memory. 
        # Loading saved models logic needs to be added if running standalone 'eval'
        print("Standalone evaluation requires loading saved models. Please run 'train' step which includes evaluation.")

if __name__ == "__main__":
    main()