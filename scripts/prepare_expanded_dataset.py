import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

input_file = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/merged_expanded_50bp.csv"
output_train = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/finetune_expanded_train.csv"
output_test = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/finetune_expanded_test.csv"

# 1. Load Data
if not os.path.exists(input_file):
    print(f"Error: Not found {input_file}")
    print("Please run the dataset.ipynb notebook to fetch the data first.")
    exit()

try:
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} positive samples from source.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Clean Data function
def clean_sequence(seq):
    seq = str(seq).upper().strip()
    seq = seq.replace('U', 'T')
    # Replace IUPAC ambiguity codes with N
    ambiguous = ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']
    for char in ambiguous:
        seq = seq.replace(char, 'N')
    # Remove any non-ACGTN characters
    seq = "".join([c for c in seq if c in "ACGTN"])
    return seq

# 3. Data Augmentation: Add Reverse Complements
# Since DNA is double-stranded, the reverse complement is physically present and valid context
print("Augmenting data with Reverse Complements...")
from Bio.Seq import Seq
augmented_seqs = []

for raw_seq in df['Expanded_Sequence']:
    cleaned = clean_sequence(raw_seq)
    
    if len(cleaned) < 30: # Skip very short ones
        continue
        
    # Add Forward
    augmented_seqs.append(cleaned)
    
    # Add Reverse Complement
    try:
        rc = str(Seq(cleaned).reverse_complement())
        augmented_seqs.append(rc)
    except:
        pass 

print(f"Total Positive Samples after Augmentation: {len(augmented_seqs)}")

# 4. Generate Negative Samples using Shuffling
# We generate 1 negative for every 1 positive to keep 50/50 balance
negative_seqs = []
max_retries = 100

print("Generating shuffled negative samples...")

for seq in augmented_seqs:
    seq_list = list(seq)
    
    # Shuffle
    random.shuffle(seq_list)
    candidate = "".join(seq_list)
    negative_seqs.append(candidate)

# 5. Create Final DataFrame
# Positives: Label 1
df_pos = pd.DataFrame({
    'text': augmented_seqs,
    'labels': 1
})

# Negatives: Label 0
df_neg = pd.DataFrame({
    'text': negative_seqs,
    'labels': 0
})

df_final = pd.concat([df_pos, df_neg], ignore_index=True)

# Shuffle all rows
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final Dataset Size: {len(df_final)} (Balanced)")

# 6. Split and Save
train, test = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['labels'])

train.to_csv(output_train, index=False)
test.to_csv(output_test, index=False)

print(f"Train set saved: {len(train)} rows -> {output_train}")
print(f"Test set saved: {len(test)} rows -> {output_test}")
