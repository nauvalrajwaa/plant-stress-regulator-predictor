import pandas as pd
import numpy as np
import os
import torch
import random
import joblib
import datasets as ds
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import sys

try:
    import einops
except ImportError:
    print("\n[CRITICAL WARNING] 'einops' library is not found. DNABERT-2 requires it!")
    print("If you are using DNABERT-2, please run: pip install einops\n")

try:
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        AutoConfig
    )
except RuntimeError as e:
    # Catching common Colab/PEFT version mismatch error
    if "modeling_layers" in str(e) or "peft" in str(e):
        print("\n" + "!"*60)
        print(" CRITICAL LIBRARY MISMATCH DETECTED")
        print("!"*60)
        print(" The installed version of 'peft' is incompatible with 'transformers'.")
        print(" This is a common issue in Google Colab environments.")
        print(" Please run the following command in a new cell to fix it:")
        print("\n    !pip install --upgrade peft accelerate transformers\n")
        print(" Then RESTART THE RUNTIME (Runtime > Restart Session) and run again.")
        print("!"*60 + "\n")
        sys.exit(1)
    raise e

# =============================================================================
# GLOBAL PATH CONFIGURATION
# =============================================================================
# Define project root (Parent of 'scripts/')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default Base Paths
BASE_BERT_PATH = os.path.join(PROJECT_ROOT, "PlantBERT")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "train", "data", "finetune_expanded_train.csv")

def predict_stress_ensemble(mined_data_path):
    """
    Trains SVM on the fly, loads PlantBERT, and predicts on the provided CSV file.
    Returns the DataFrame with predictions.
    """
    print(f"--- STARTING ENSEMBLE PREDICTION ---")
    print(f"Target Input Data: {mined_data_path}")

    # 0. Validate Inputs
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[ERROR] Training data not found at: {TRAIN_DATA_PATH}")
        return None
    if not os.path.exists(mined_data_path):
        print(f"[ERROR] Mined data not found: {mined_data_path}")
        return None

    # Load Data
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_mined = pd.read_csv(mined_data_path)
    
    if 'Extracted_Sequence' not in df_mined.columns:
        print("Column 'Extracted_Sequence' missing from input data.")
        return None
    
    # ---------------------------------------------------------
    # MODEL 1: SVM (The Benchmark Winner) - Training on the Fly
    # ---------------------------------------------------------
    print("\n[1/4] Training SVM Model...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(df_train['text'])
    y_train = df_train['labels']
    
    svm_model = SVC(C=10, kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Predict SVM
    X_new = vectorizer.transform(df_mined['Extracted_Sequence'])
    svm_probs = svm_model.predict_proba(X_new)[:, 1]
    print("      -> SVM Predictions done.")

    # ---------------------------------------------------------
    # MODEL 2: PLANTBERT (Deep Learning) - Loading from Disk
    # ---------------------------------------------------------
    print("\n[2/4] Loading PlantBERT Model...")
    try:
        # Try loading fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        print("      -> Loaded Fine-Tuned PlantBERT.")
    except OSError:
        print("      -> Fine-tuned model not found. Using Base PlantBERT (Untrained for this task - WARNING).")
        tokenizer = AutoTokenizer.from_pretrained(BASE_BERT_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_BERT_PATH, num_labels=2)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"      -> Running inference on: {device.upper()}")
    model.to(device)
    model.eval()
    
    print(f"      -> Predicting with PlantBERT...")
    bert_probs = []
    texts = df_mined['Extracted_Sequence'].tolist()
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Max length for BERT allows for 512, but our sequences are short (~100bp)
            if callable(tokenizer):
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=150)
            else:
                 inputs = tokenizer.batch_encode_plus(
                    batch, max_length=150, pad_to_max_length=True, truncation=True, return_tensors='pt'
                 )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Class 1 is 'Positive'
            bert_probs.extend(probs[:, 1].cpu().numpy())
            
            print(f"         Processed {min(i+batch_size, len(texts))}/{len(texts)}...", end='\r')
            
    bert_probs = np.array(bert_probs)
    print("\n      -> PlantBERT Predictions done.")

    # ---------------------------------------------------------
    # ENSEMBLE AGGREGATION
    # ---------------------------------------------------------
    print("\n[3/4] Calculating Ensemble Scores...")
    # Weighted Average: SVM (0.6) + BERT (0.4) 
    ensemble_probs = (0.6 * svm_probs) + (0.4 * bert_probs)
    
    df_mined['SVM_Prob'] = svm_probs.round(4)
    df_mined['BERT_Prob'] = bert_probs.round(4)
    df_mined['Stress_Probability'] = ensemble_probs.round(4)
    
    # Threshold > 0.5 is Positive
    df_mined['Predicted_Label'] = (df_mined['Stress_Probability'] > 0.5).astype(int)
    df_mined['Prediction_Status'] = df_mined['Predicted_Label'].apply(lambda x: 'STRESS REGION' if x==1 else 'Neutral')

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    print("\n[4/4] Saving Results...")
    high_conf = df_mined[df_mined['Stress_Probability'] > 0.8].copy()
    
    result_filename = "Final_Predictions.csv" # Overwrite previous file
    df_mined.to_csv(result_filename, index=False)
    
    print(f"      -> Results saved to: {result_filename}")
    print(f"      -> High Confidence Candidates (>80%): {len(high_conf)} found.")
    
    if not high_conf.empty:
        print("\n--- TOP CANDIDATES (Ensemble) ---")
        cols = ['Gene_Locus', 'Motif_ID', 'SVM_Prob', 'BERT_Prob', 'Stress_Probability']
        print(high_conf[cols].head(10).to_string(index=False))
        
    return df_mined

def train_new_model_from_mined_data(mined_data_path, output_model_path="svm_retrained_mined.pkl"):
    """
    Trains a NEW SVM model using the mined data as Label 1 (Positive) 
    and generates Random Sequences as Label 0 (Negative).
    """
    print(f"--- STARTING RETRAINING FROM MINED DATA ---")
    print(f"Source Data: {mined_data_path}")
    
    # 1. Load Mined Data (Positives)
    if not os.path.exists(mined_data_path):
        print(f"[ERROR] File not found: {mined_data_path}")
        return None, None
        
    df_mined = pd.read_csv(mined_data_path)
    # Ensure we have sequences
    if 'Extracted_Sequence' not in df_mined.columns:
        print("[ERROR] Column 'Extracted_Sequence' is missing.")
        return None, None
        
    positive_seqs = df_mined['Extracted_Sequence'].tolist()
    positive_labels = [1] * len(positive_seqs)
    
    print(f"      -> Loaded {len(positive_seqs)} positive sequences (from mining).")
    
    # 2. Generate Random Data (Negatives)
    # matching the count and approx length of positives
    print(f"      -> Generating {len(positive_seqs)} random negative sequences...")
    avg_len = int(sum(len(s) for s in positive_seqs) / len(positive_seqs))
    
    negative_seqs = []
    bases = ['A', 'C', 'G', 'T']
    for _ in range(len(positive_seqs)):
        # Generate random length close to average (+/- 20%)
        length = random.randint(int(avg_len*0.8), int(avg_len*1.2))
        seq = "".join(random.choices(bases, k=length))
        negative_seqs.append(seq)
    
    negative_labels = [0] * len(negative_seqs)
    
    # 3. Combine
    all_seqs = positive_seqs + negative_seqs
    all_labels = positive_labels + negative_labels
    
    # 4. Vectorize
    print(f"      -> Vectorizing data (3-5 gram)...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5))
    X = vectorizer.fit_transform(all_seqs)
    y = np.array(all_labels)
    
    # 5. Train SVM
    print(f"      -> Training SVM (RBF Kernel)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm_model = SVC(C=10, kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Val Accuracy
    preds = svm_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"      -> Retraining Complete. Validation Accuracy: {acc*100:.2f}%")
    

def prepare_augmented_data(mined_data_path, task_type="binary"):
    """
    Helper to load mined data.
    If task_type='binary': Generates random negatives to match positives (Label 1).
    If task_type='multiclass': Uses existing 'Label' column from CSV.
    """
    # 1. Load Data
    if not os.path.exists(mined_data_path):
        print(f"[ERROR] File not found: {mined_data_path}")
        return None, None
        
    df_mined = pd.read_csv(mined_data_path)
    if 'Extracted_Sequence' not in df_mined.columns:
        print("[ERROR] Column 'Extracted_Sequence' is missing.")
        return None, None

    if task_type == "multiclass":
        if 'Label' not in df_mined.columns:
             print("[ERROR] Multiclass mode requires 'Label' column in input CSV.")
             return None, None
        
        print(f"      -> Multiclass Mode: Found labels {df_mined['Label'].unique()}")
        df_full = df_mined.rename(columns={'Extracted_Sequence': 'text', 'Label': 'labels'})
        
        # Ensure labels are integers 0..N
        # If labels are strings, encode them
        if df_full['labels'].dtype == 'O':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_full['labels'] = le.fit_transform(df_full['labels'])
            print(f"      -> Encoded labels mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    else:
        # Binary Mode: Generate Negatives
        positive_seqs = df_mined['Extracted_Sequence'].tolist()
        positive_labels = [1] * len(positive_seqs)
        
        # Generate Random Data (Negatives)
        avg_len = int(sum(len(s) for s in positive_seqs) / len(positive_seqs))
        
        negative_seqs = []
        bases = ['A', 'C', 'G', 'T']
        for _ in range(len(positive_seqs)):
            length = random.randint(int(avg_len*0.8), int(avg_len*1.2))
            seq = "".join(random.choices(bases, k=length))
            negative_seqs.append(seq)
        
        negative_labels = [0] * len(negative_seqs)
        
        # Combine
        all_seqs = positive_seqs + negative_seqs
        all_labels = positive_labels + negative_labels
        
        df_full = pd.DataFrame({'text': all_seqs, 'labels': all_labels})
    
    # 4. Split
    train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['labels'])
    return train_df, test_df

def train_multimodel_ml(mined_data_path, output_dir="models", organism="Unknown", task_type="binary", save_model=True):
    """
    Trains Multiple ML Models (SVM, RF, GB, LR) on the mined data + random negatives.
    Performs Grid Search to find the best hyperparameters.
    Returns the BEST model and vectorizer.
    """
    print(f"\n--- STARTING MULTI-MODEL ML TRAINING ({task_type.upper()}) ---")
    
    # 1. Prepare Data
    train_df, test_df = prepare_augmented_data(mined_data_path, task_type)
    if train_df is None: return None, None
    print(f"      -> Data split: {len(train_df)} Train, {len(test_df)} Test")

    # 2. Vectorize
    print(f"      -> Feature Engineering (3-5 gram char vectorizer)...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['labels']
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['labels']

    # 3. Define Models & Grid
    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1, 10], "solver": ["liblinear"]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3, 5]}
        }
    }
    
    results = []
    best_overall_acc = 0
    best_overall_model = None
    best_model_name = ""

    # 4. Train Loop
    for model_name, config in models_config.items():
        print(f"\n      -> Training {model_name}...")
        start_t = time.time()
        
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_t
        
        print(f"         Best Params: {grid.best_params_}")
        print(f"         Test Accuracy: {acc*100:.2f}% ({elapsed:.1f}s)")
        
        results.append({"Model": model_name, "Accuracy": acc, "Best Params": str(grid.best_params_)})
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_model = best_model
            best_model_name = model_name

    # 5. Summary
    print("\n      --- MODEL LEADERBOARD ---")
    res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(res_df.to_string(index=False))
    
    print(f"\n      -> 🏆 WINNER: {best_model_name} (Acc: {best_overall_acc*100:.2f}%)")
    
    # 6. Save (Updated Logic)
    if save_model and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Format: {model_name}_{accuracy}_{organism}.pkl
        acc_str = f"{best_overall_acc*100:.1f}"
        org_clean = organism.replace(" ", "_")
        
        filename = f"{best_model_name}_{acc_str}_{org_clean}.pkl"
        output_path = os.path.join(output_dir, filename)
        
        print(f"      -> Saving winner to: {output_path}")
        joblib.dump(best_overall_model, output_path)
        joblib.dump(vectorizer, output_path.replace(".pkl", "_vectorizer.pkl"))
    else:
        print("      -> Saving skipped (flag disabled).")
    
    return best_overall_model, vectorizer


def train_plantbert_from_mined_data(mined_data_path, output_dir="models", organism="Unknown", task_type="binary", save_model=True, base_model_path=None, kmer=6):
    """
    Trains PlantBERT using the mined data + random negatives.
    Replicates the logic from finetune_plantbert_expanded.py but callable.
    """
    print(f"\n--- STARTING TRANSFORMER RETRAINING ({task_type.upper()}) ---")
    
    # Use provided base path or default global
    # FIX: Use global BASE_BERT_PATH if argument is None
    model_source_path = base_model_path if base_model_path else BASE_BERT_PATH
    print(f"      -> Base Model Source: {model_source_path}")
    
    # 1. Prepare Data
    train_df, test_df = prepare_augmented_data(mined_data_path, task_type)
    if train_df is None: return None, None
    
    print(f"      -> Data split: {len(train_df)} Train, {len(test_df)} Test")
    
    # Determine number of labels
    num_labels = len(train_df['labels'].unique())
    print(f"      -> Detected {num_labels} classes.")
    
    # ------------------------------------------------------------------
    # SPECIALIZED ROUTE: DNABERT-2 (Using Notebook Wrapper)
    # ------------------------------------------------------------------
    if "dnabert" in model_source_path.lower() and "zhihan1996" in model_source_path.lower():
        print("\n      -> [Pipeline] Redirecting to Specialized DNABERT-2 Module...")
        try:
            # Dynamic import to handle dependency
            import sys
            sys.path.append(os.path.dirname(__file__)) # Ensure scripts folder is in path
            import dnabert2_finetuner
            
            # Prepare Temporary CSVs (Notebook expects CSV input)
            models_dir_abs = os.path.abspath(output_dir) if output_dir else os.path.join(os.getcwd(), "models")
            os.makedirs(models_dir_abs, exist_ok=True)
            
            # Rename columns to match DNABERT-2 expectations: [sequence, label]
            train_tmp_path = os.path.join(models_dir_abs, "temp_train.csv")
            test_tmp_path = os.path.join(models_dir_abs, "temp_test.csv")
            
            train_df.rename(columns={'text': 'sequence', 'labels': 'label'}).to_csv(train_tmp_path, index=False)
            test_df.rename(columns={'text': 'sequence', 'labels': 'label'}).to_csv(test_tmp_path, index=False)
            
            # Run Finetuner
            # Determine batch size dynamically
            # DNABERT-2 can be heavy. Use small batch.
            use_lora = False # Or make configurable? User didn't specify. Notebook default False.
            
            final_model, final_tokenizer = dnabert2_finetuner.run_dnabert2_finetuning(
                train_csv_path=train_tmp_path,
                val_csv_path=test_tmp_path,
                output_dir=models_dir_abs,
                model_name_or_path=model_source_path,
                epochs=3,
                batch_size=8, # Adjusted as per notebook
                save_model=save_model,
                use_lora=use_lora
            )
            
            return final_model, final_tokenizer
            
        except ImportError as e:
            print(f"[ERROR] Could not import dnabert2_finetuner: {e}")
            print("Falling back to standard generic trainer...")
        except Exception as e:
            print(f"[ERROR] DNABERT-2 Fine-tuning failed: {e}")
            print("Falling back to generic trainer...")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # SPECIALIZED ROUTE: DNABERT-1 (Using Notebook Wrapper)
    # ------------------------------------------------------------------
    # Check for "DNA_bert_X" pattern specific to v1
    if "dna_bert" in model_source_path.lower():
        print("\n      -> [Pipeline] Redirecting to Specialized DNABERT-1 Module...")
        try:
            import sys
            sys.path.append(os.path.dirname(__file__)) 
            import dnabert1_finetuner
            
            models_dir_abs = os.path.abspath(output_dir) if output_dir else os.path.join(os.getcwd(), "models")
            os.makedirs(models_dir_abs, exist_ok=True)
            
            # Export Temp Data
            train_tmp_path = os.path.join(models_dir_abs, "temp_dnabert1_train.csv")
            test_tmp_path = os.path.join(models_dir_abs, "temp_dnabert1_test.csv")
            
            # Use columns [sequence, label]
            train_df.rename(columns={'text': 'sequence', 'labels': 'label'}).to_csv(train_tmp_path, index=False)
            test_df.rename(columns={'text': 'sequence', 'labels': 'label'}).to_csv(test_tmp_path, index=False)
            
            # Extract K-mer from model name or use argument
            import re
            kmer_match = re.search(r"dna_bert_(\d)", model_source_path.lower())
            
            # Logic: If argument kmer is valid (3-6), use it. Else try regex. Else default 6.
            if kmer in [3, 4, 5, 6]:
                kmer_k = kmer
            elif kmer_match:
                kmer_k = int(kmer_match.group(1))
            else:
                kmer_k = 6
                
            print(f"      -> Using K-mer: {kmer_k}")

            final_model, final_tokenizer = dnabert1_finetuner.run_dnabert1_finetuning(
                train_csv_path=train_tmp_path,
                val_csv_path=test_tmp_path,
                output_dir=models_dir_abs,
                kmer=kmer_k,
                epochs=3,
                batch_size=16, # DNABERT-1 is lighter than V2
                save_model=save_model,
                model_name_or_path=model_source_path
            )
            
            return final_model, final_tokenizer
            
        except ImportError as e:
            print(f"[ERROR] Could not import dnabert1_finetuner: {e}")
        except Exception as e:
            print(f"[ERROR] DNABERT-1 Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()


    # Convert to HF Dataset
    dataset = ds.DatasetDict({
        "train": ds.Dataset.from_pandas(train_df),
        "test": ds.Dataset.from_pandas(test_df)
    })
    

    # 2. Setup Model & Tokenizer
    # FIX: Always load tokenizer from the Base path to avoid missing file errors in checkpoints
    print(f"      -> Loading Tokenizer from: {model_source_path}")
    try:
        # Trust remote code is sometimes needed for DNABERT-2 or custom models
        tokenizer = AutoTokenizer.from_pretrained(model_source_path, trust_remote_code=True)
    except OSError:
        # If user hasn't downloaded it yet, let it download or fail
        print(f"[WARNING] Local base tokenizer not found at {model_source_path}. Trying 'bert-base-uncased' as fallback...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    # Check model max position embeddings to avoid shape errors
    # Standard PlantBERT/BERT usually 128 or 512.
    # DNABERT-2 can handle longer sequences (ALiBi).
    is_dnabert = "dnabert" in model_source_path.lower()
    is_agront = "agro" in model_source_path.lower() or "nucleotide" in model_source_path.lower()
    
    if is_dnabert:
        # DNABERT-2 supports long context. Let's start with 512 but allow up to 1024 if the tokenizer supports it.
        # We cap at 1024 for memory efficiency during finetuning on moderate GPUs.
        default_cap = 1024
    elif is_agront:
        # Agro-NT models are large (1B params). Context length usually 1000+ but constrained by GPU memory.
        # Standard NT is 1000bp.
        default_cap = 1024
    else:
        # PlantBERT / Standard BERT usually 512 max
        default_cap = 512

    if hasattr(tokenizer, 'model_max_length'):
        safe_max = tokenizer.model_max_length
        if safe_max > 100000: # Handle 'very large' arbitrary values like int.max
            max_seq_len = default_cap
        else:
            max_seq_len = min(default_cap, safe_max)
    else:
        max_seq_len = 128 # Safe default for smaller/older models
        
    print(f"      -> Flexible Shape Strategy: {'DNABERT (ALiBi)' if is_dnabert else ('AgroNT (Large)' if is_agront else 'PlantBERT (PosEmbed)')}")
    print(f"      -> Using max sequence length: {max_seq_len}")

    def tokenize_function(examples):
        # Explicit verify tokenizer callable (Legacy Transformers v2.x support)
        if not callable(tokenizer):
            # Check for batch_encode_plus (standard in v2.x)
            if hasattr(tokenizer, "batch_encode_plus"):
                return tokenizer.batch_encode_plus(
                    examples["text"], 
                    max_length=max_seq_len, 
                    pad_to_max_length=True,
                    truncation=True
                )
            
            # If tokenizer is broken/None, try a safe fallback
            print("[CRITICAL WARNING] Tokenizer object provided is not callable/compatible. Regenerating...")
            from transformers import AutoTokenizer
            safe_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Fallback usage
            if not callable(safe_tok) and hasattr(safe_tok, "batch_encode_plus"):
                 return safe_tok.batch_encode_plus(
                    examples["text"], 
                    max_length=max_seq_len, 
                    pad_to_max_length=True,
                    truncation=True
                 )
            return safe_tok(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)
            
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Handle potential column differences
    cols_to_remove = [c for c in ["text", "__index_level_0__"] if c in tokenized_datasets["train"].column_names]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
    tokenized_datasets.set_format("torch")
    
    # Load Model
    print(f"      -> Loading Model Weights from: {model_source_path}")
    try:
        # DNABERT-2 / Custom Model Debugging
        if "DNABERT" in model_source_path or "dnabert" in model_source_path.lower():
             # ... (DNABERT LOGIC REMOVED FOR BREVITY in display, but kept in execution) ...
             print("      -> Detected DNABERT model. Verifying dependencies...")
             
             # Attempt to pre-validate custom code loading to expose hidden import errors
             bert_layers_path = os.path.join(model_source_path, "bert_layers.py")
             if os.path.isdir(model_source_path) and os.path.exists(bert_layers_path):
                 import sys, importlib.util
                 try:
                     spec = importlib.util.spec_from_file_location("bert_layers_check", bert_layers_path)
                     if spec and spec.loader:
                         module = importlib.util.module_from_spec(spec)
                         spec.loader.exec_module(module)
                         print("      -> 'bert_layers.py' is valid and loadable.")
                         
                     # Triton Check
                     try:
                        import triton
                        print("      -> Triton detected. If you see 'trans_b' errors, it means a version mismatch.")
                     except ImportError:
                        pass
                 except Exception as import_err:
                     print(f"\n      -> [CRITICAL ERROR] Custom model code failed to load: {import_err}")
                     raise import_err

             model = AutoModelForSequenceClassification.from_pretrained(
                model_source_path, 
                num_labels=num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )

             # Monkey Patch to disable buggy Flash Attention if model has it
             if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
                  for layer in model.bert.encoder.layer:
                       if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                            if hasattr(layer.attention.self, "use_flash_attention"):
                                 print("      -> [Auto-Fix] Disabling Flash Attention (Triton) to prevent 'trans_b' crash.")
                                 layer.attention.self.use_flash_attention = False
                                 
        elif "agro" in model_source_path.lower() or "nucleotide" in model_source_path.lower():
             # AGRO-NT Logic
             print("      -> Detected Agro-NT / Nucleotide Transformer.")
             # These models often require trust_remote_code=True as well.
             # They are usually MaskedLM-based, so loading for SequenceClassification adds a new head.
             # WARNING: 1B+ params. might OOM on small GPUs.
             
             model = AutoModelForSequenceClassification.from_pretrained(
                model_source_path, 
                num_labels=num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
             
             # Verify if Ebeddings need resizing (AgroNT usually has large vocab)
             model.resize_token_embeddings(len(tokenizer))
             
        else:
            # Standard BERT / PlantBERT
            from transformers import AutoConfig
            
            # Robust Config Loading (Legacy vs Modern)
            try:
                config = AutoConfig.from_pretrained(model_source_path, num_labels=num_labels)
            except Exception:
                 # Try with trust_remote_code if supported (newer transformers)
                 try:
                    config = AutoConfig.from_pretrained(model_source_path, num_labels=num_labels, trust_remote_code=True)
                 except TypeError:
                    config = AutoConfig.from_pretrained(model_source_path, num_labels=num_labels)

            # Robust Model Loading
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_source_path, 
                    config=config,
                    ignore_mismatched_sizes=True,
                    trust_remote_code=True
                )
            except (TypeError, ValueError):
                # Retry for Legacy Transformers (v2.x - v4.x)
                print("      -> [Compatibility] Retrying load without newer args (ignore_mismatched_sizes, etc)...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_source_path, 
                    config=config
                )
            
            # Critical Fix for "Size of tensor a (512) must match size of tensor b (128)"
            # Some older PlantBERT models have max_position_embeddings=128 but we tokenized to 512.
            # We must resize the positional embeddings if they are too small.
            if hasattr(model, "bert") and hasattr(model.bert.embeddings, "position_embeddings"):
                current_max_pos = model.config.max_position_embeddings
                if current_max_pos < max_seq_len:
                    print(f"      -> [Auto-Fix] Resizing model position embeddings from {current_max_pos} to {max_seq_len}")
                    
                    # 1. Update Config first
                    model.config.max_position_embeddings = max_seq_len
                    
                    # 2. Get old embeddings
                    old_pos_embeddings = model.bert.embeddings.position_embeddings
                    old_weights = old_pos_embeddings.weight.data
                    
                    # 3. Create new embedding layer
                    new_pos_embeddings = torch.nn.Embedding(max_seq_len, model.config.hidden_size)
                    
                    # 4. Copy weights (repeat/cycle them if needed, or just partial copy)
                    # Safe copy: copy what we can, leave rest as random init or copy last
                    # Standard approach: Copy existing, then initialize new ones randomly or by copying
                    n = min(old_weights.shape[0], max_seq_len) # Ensure fit
                    new_pos_embeddings.weight.data[:n, :] = old_weights[:n, :]
                    
                    # 5. Assign back to model
                    model.bert.embeddings.position_embeddings = new_pos_embeddings
                    
                    # 6. IMPORTANT: Update token type ids buffer if it exists (common source of errors)
                    if hasattr(model.bert.embeddings, "token_type_ids"):
                         # Re-register buffer with correct size
                         model.bert.embeddings.register_buffer("token_type_ids", 
                             torch.zeros((1, max_seq_len), dtype=torch.long), persistent=False)
                    
                    # 7. Update positional ids buffer (often overlooked)
                    if hasattr(model.bert.embeddings, "position_ids"):
                         model.bert.embeddings.register_buffer("position_ids",
                             torch.arange(max_seq_len).expand((1, -1)), persistent=False)
            
            # Additional check for general mismatch (sometimes helps with other architectures)
            if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
                 if model.config.max_position_embeddings < max_seq_len:
                      print(f"      -> [Warning] Model config {model.config.max_position_embeddings} < {max_seq_len}. Resizing might have been needed but skipped if not standard BERT.")

    except Exception as e:
        print(f"      -> [ERROR] Failed to load {model_source_path}: {e}")
        return None, None

    model.resize_token_embeddings(len(tokenizer))

    
    # 3. Metrics
    if EVALUATE_AVAILABLE:
        metric = evaluate.load("accuracy")
        
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        if EVALUATE_AVAILABLE:
             return metric.compute(predictions=predictions, references=labels)
        else:
             return {"accuracy": accuracy_score(labels, predictions)}

    # 4. Trainer
    # Disable W&B logging explicitly
    os.environ["WANDB_DISABLED"] = "true"

    # Directory logic
    # If output_dir is provided (e.g., runs/ID/models), place checkpoints in runs/ID/checkpoints
    # Otherwise fallback to global models/plantbert_checkpoints
    if output_dir and "runs" in output_dir:
        # Assuming structure runs/ID/models -> go up to runs/ID -> add checkpoints
        run_root = os.path.dirname(output_dir)
        ckpt_dir = os.path.join(run_root, "checkpoints")
    else:
        ckpt_dir = os.path.join(PROJECT_ROOT, "models", "plantbert_checkpoints")
    
    print(f"      -> Checkpoints will be saved to: {ckpt_dir}")

    # Dynamic Hyperparams based on Model Size
    # Re-evaluate flags if needed, or rely on earlier definitions if in scope.
    is_dnabert = "dnabert" in model_source_path.lower()
    is_agront = "agro" in model_source_path.lower() or "nucleotide" in model_source_path.lower()

    if is_agront:
        # Agro-NT (1B params) requires massive memory saving
        batch_size = 1 
        grad_acc = 16   # effective batch ~16
        use_fp16 = torch.cuda.is_available() # Use Mixed Precision if CUDA
        use_grad_ckpt = True # Essential for 1B model to trade compute for memory
        print("      -> [Config] Agro-NT detected: Using BatchSize=1, GradAccum=16, GradCheckpoint=True, FP16={use_fp16}")
    elif is_dnabert:
        # DNABERT-2 (117M) but complex attention
        batch_size = 4
        grad_acc = 4
        use_fp16 = torch.cuda.is_available()
        use_grad_ckpt = False # Flash Attention usually handles it, but safety first
        print(f"      -> [Config] DNABERT detected: Using BatchSize={batch_size}, GradAccum={grad_acc}, FP16={use_fp16}")
    else:
        # Standard PlantBERT (small)
        # Check for MPS (Apple Silicon) safely (PyTorch 1.7.1 compat)
        try:
             is_mps = torch.backends.mps.is_available()
        except AttributeError:
             is_mps = False
             
        batch_size = 8 if is_mps else 16
        grad_acc = 2 if is_mps else 1
        use_fp16 = False # MPS doesn't fully stabilize with fp16 in Trainer sometimes
        use_grad_ckpt = False
        print(f"      -> [Config] Standard Model: Using BatchSize={batch_size}")

    # Prepare Training Arguments with version compatibility
    import transformers
    major_ver = int(transformers.__version__.split('.')[0])
    
    if major_ver < 3: # Legacy (v2.x) - strict for DNABERT1 envs
        # Manual calculation for per_gpu vs per_device
        # v2.11 uses per_gpu_train_batch_size
        training_args = TrainingArguments(
            output_dir=ckpt_dir,
            overwrite_output_dir=True,
            evaluate_during_training=True, # v2.x legacy
            num_train_epochs=3,
            per_gpu_train_batch_size=batch_size,
            per_gpu_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_acc,
            fp16=False, # Apex likely missing
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
        )
    else:
        # Modern (v3.x - v4.x+)
        # 'eval_strategy' is very new (v4.41+), 'evaluation_strategy' is standard v4.x
        strategy_arg = "epoch"
        kwargs = {
            "output_dir": ckpt_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": grad_acc,
            "fp16": use_fp16, # keep user pref for modern envs
            "gradient_checkpointing": use_grad_ckpt,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "logging_steps": 10,
            "report_to": "none",
            "save_strategy": "epoch"
        }
        
        # Safe strategy arg
        try:
             # Try modern
             TrainingArguments(output_dir="tmp", eval_strategy="epoch")
             kwargs["eval_strategy"] = "epoch"
        except TypeError:
             # Fallback
             kwargs["evaluation_strategy"] = "epoch"

        training_args = TrainingArguments(**kwargs)
    
    # Define legacy collator for v2.x (which expects objs with .collate_batch)
    class LegacyDictCollator:
        def collate_batch(self, features):
            import torch
            batch = {}
            first = features[0]
            for k in first.keys():
                if k == "label" or k == "labels":
                     # Ensure labels are LongTensor key 'labels'
                     batch["labels"] = torch.stack([f[k] for f in features])
                else:
                     batch[k] = torch.stack([f[k] for f in features])
            return batch

    # Trainer (Compatibility)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["test"],
        "compute_metrics": compute_metrics,
        "data_collator": LegacyDictCollator(), 
    }
    
    # Newer transformers support passing 'tokenizer' to Trainer for auto-saving
    # Older ones (v2.x) do not have this argument in __init__
    import inspect
    init_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in init_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = Trainer(**trainer_kwargs)
    
    print(f"      -> Starting Training (this may take a while)...")
    trainer.train()
    
    # Eval
    metrics = trainer.evaluate()
    final_acc = metrics['eval_accuracy']
    print(f"      -> Final Evaluation: {metrics}")
    
    if save_model and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Clean naming for folder
        acc_str = f"{final_acc*100:.1f}"
        org_clean = organism.replace(" ", "_")
        folder_name = f"PlantBERT_{acc_str}_{org_clean}"
        full_path = os.path.join(output_dir, folder_name)
        
        print(f"      -> Saving PlantBERT to {full_path}")
        trainer.save_model(full_path)
        tokenizer.save_pretrained(full_path)
    else:
        print("      -> Saving skipped.")
    
    return trainer.model, tokenizer

def predict_with_new_models(target_csv, svm_model=None, vectorizer=None, bert_model=None, bert_tokenizer=None):
    """
    Runs prediction using the RECENTLY TRAINED models passed directly from memory.
    """
    print(f"\n--- RUNNING FINAL ENSEMBLE PREDICTION ---")
    
    # 1. Load Data
    if not os.path.exists(target_csv):
        print("Target CSV not found.")
        return None
    df = pd.read_csv(target_csv)
    if 'Extracted_Sequence' not in df.columns:
        return None
    
    seqs = df['Extracted_Sequence'].tolist()
    
    # 2. SVM Prediction
    print("      -> getting ML probabilities...")
    if svm_model and vectorizer:
        X_new = vectorizer.transform(seqs)
        svm_probs = svm_model.predict_proba(X_new)[:, 1]
    else:
        print("[Warn] No SVM model provided. Using 0.5 neutral prob.")
        svm_probs = np.full(len(seqs), 0.5)

    # 3. PlantBERT Prediction
    print("      -> getting PlantBERT probabilities...")
    bert_probs = []
    
    if bert_model and bert_tokenizer:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"         [INFO] Using device: {device}")
        bert_model.to(device)
        bert_model.eval()
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i:i+batch_size]
                
                if callable(bert_tokenizer):
                    inputs = bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                else:
                    inputs = bert_tokenizer.batch_encode_plus(
                         batch, max_length=128, pad_to_max_length=True, truncation=True, return_tensors='pt'
                    )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = bert_model(**inputs)
                # Compatibility: Legacy vs Modern Transformers
                # Modern: outputs.logits
                # Legacy: outputs[0] (tuple)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs 
                    
                probs = torch.nn.functional.softmax(logits, dim=-1)
                bert_probs.extend(probs[:, 1].cpu().numpy())
    else:
        print("[Warn] No PlantBERT provided. Using 0.5 neutral prob.")
        bert_probs = np.full(len(seqs), 0.5)
        
    bert_probs = np.array(bert_probs)

    # 4. Ensemble
    # weighted average (ML 50% + DeepLearning 50%)
    ensemble_probs = (0.5 * svm_probs) + (0.5 * bert_probs)
    
    df['ML_Prob'] = svm_probs.round(4)
    df['BERT_Prob'] = bert_probs.round(4)
    df['Stress_Probability'] = ensemble_probs.round(4)
    df['Prediction'] = (df['Stress_Probability'] > 0.5).astype(int)
    
    # 5. Save
    out_file = "Final_Ensemble_Predictions.csv"
    df.to_csv(out_file, index=False)
    print(f"      -> Saved to {out_file}")
    
    return df


def evaluate_models_on_holdout(mined_data_path, svm_model, vectorizer, bert_model, bert_tokenizer):
    """
    Re-creates the Train/Test split (random_state=42) and evaluates the current models 
    on the Test partition to report detailed performance metrics.
    """
    print(f"\n--- COMPREHENSIVE MODEL EVALUATION (20% Hold-out Test Set) ---")
    train_df, test_df = prepare_augmented_data(mined_data_path)
    
    metrics = {}
    
    # 1. Evaluate ML Model
    if svm_model and vectorizer:
        print("\n>>> 1. ML MODEL PERFORMANCE (Scanning...)")
        X_test = vectorizer.transform(test_df['text'])
        y_test = test_df['labels']
        y_pred = svm_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        metrics['ML_Accuracy'] = acc
        
        print(f"    • Accuracy: {acc*100:.2f}%")
        print("    • Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        print("    • Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    # 2. Evaluate PlantBERT
    if bert_model and bert_tokenizer:
        print("\n>>> 2. PLANTBERT PERFORMANCE (Scanning...)")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        bert_model.to(device)
        bert_model.eval()
        
        preds = []
        labels = test_df['labels'].tolist()
        texts = test_df['text'].tolist()
        
        # Batch inference
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_text = texts[i:i+batch_size]
                
                if callable(bert_tokenizer):
                    inputs = bert_tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                else:
                    inputs = bert_tokenizer.batch_encode_plus(
                        batch_text, max_length=128, pad_to_max_length=True, truncation=True, return_tensors='pt'
                    )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = bert_model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                preds.extend(batch_preds)
        
        acc = accuracy_score(labels, preds)
        metrics['BERT_Accuracy'] = acc
        
        print(f"    • Accuracy: {acc*100:.2f}%")
        print("    • Classification Report:")
        print(classification_report(labels, preds, target_names=['Negative', 'Positive']))
        
        print("    • Confusion Matrix:")
        print(confusion_matrix(labels, preds))

    return metrics

