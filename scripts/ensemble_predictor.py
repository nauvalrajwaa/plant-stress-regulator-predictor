import pandas as pd
import numpy as np
import os
import torch
import random
import joblib
import datasets as ds
import evaluate
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
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=150)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logits = model(**inputs).logits
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


def train_plantbert_from_mined_data(mined_data_path, output_dir="models", organism="Unknown", task_type="binary", save_model=True, base_model_path=None):
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
        
    def tokenize_function(examples):
        # Increased max_length to 512 for better context if model supports it
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
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
             print("      -> Detected DNABERT model. Verifying dependencies...")
             
             # Attempt to pre-validate custom code loading to expose hidden import errors
             bert_layers_path = os.path.join(model_source_path, "bert_layers.py")
             if os.path.isdir(model_source_path) and os.path.exists(bert_layers_path):
                 import sys, importlib.util
                 try:
                     spec = importlib.util.spec_from_file_location("bert_layers_check", bert_layers_path)
                     if spec and spec.loader:
                         module = importlib.util.module_from_spec(spec)
                         # We don't register it in sys.modules to avoid conflict, just execute to test
                         spec.loader.exec_module(module)
                         print("      -> 'bert_layers.py' is valid and loadable.")
                 except Exception as import_err:
                     print(f"\n      -> [CRITICAL ERROR] Custom model code failed to load: {import_err}")
                     print(f"         Make sure you have installed: einops")
                     print(f"         This failure causes Transformers to fallback to standard BERT, leading to Config Mismatches.\n")
                     # We assume trust_remote_code will fail similarly, but let it try or just raise here
                     raise import_err

             model = AutoModelForSequenceClassification.from_pretrained(
                model_source_path, 
                num_labels=num_labels,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
        else:
            # Standard BERT / PlantBERT
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_source_path, trust_remote_code=True, num_labels=num_labels)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source_path, 
                config=config,
                ignore_mismatched_sizes=True,
                trust_remote_code=True
            )
    except Exception as e:
        print(f"      -> [ERROR] Failed to load {model_source_path}: {e}")
        return None, None

    model.resize_token_embeddings(len(tokenizer))

    
    # 3. Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    # 4. Trainer
    # Temp dir for checkpoints, final save is controlled below
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "plantbert_checkpoints")
    
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3, # Reduced slightly for interactive speed, user can increase
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
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
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        bert_model.to(device)
        bert_model.eval()
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i:i+batch_size]
                inputs = bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = bert_model(**inputs).logits
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
        device = "mps" if torch.backends.mps.is_available() else "cpu"
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
                inputs = bert_tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = bert_model(**inputs).logits
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

