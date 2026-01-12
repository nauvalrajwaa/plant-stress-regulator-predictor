import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

input_file = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/merged_place_seq.csv"
output_train = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/finetune_train.csv"
output_test = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/finetune_test.csv"

# 1. Load Data
try:
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} positive samples.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Clean Data
def clean_sequence(seq):
    seq = str(seq).upper().strip()
    # Replace common RNA -> DNA
    seq = seq.replace('U', 'T')
    # Replace IUPAC ambiguity codes with N (conservative approach)
    # R, Y, S, W, K, M, B, D, H, V
    ambiguous = ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']
    for char in ambiguous:
        seq = seq.replace(char, 'N')
    # Remove any non-ACGTN characters
    seq = "".join([c for c in seq if c in "ACGTN"])
    return seq

df['clean_seq'] = df['Sequence'].apply(clean_sequence)
# Remove empty sequences
df = df[df['clean_seq'].str.len() > 0]
print(f"Retained {len(df)} samples after cleaning.")

# 3. Generate Negative Samples using Shuffling (Preserves Length & %GC)
negative_seqs = []
max_retries = 100
bases = ['A', 'C', 'G', 'T']
positive_seq_set = set(df['clean_seq'].unique())

for seq in df['clean_seq']:
    seq_list = list(seq)
    found = False
    
    # Try shuffling first (preserves exact composition)
    for _ in range(max_retries):
        random.shuffle(seq_list)
        candidate = "".join(seq_list)
        
        # Criteria for a valid negative:
        # 1. Must not be identical to the source sequence (essential!)
        # 2. Must not match ANY known positive sequence (prevents accidental creation of real motifs)
        if candidate != seq and candidate not in positive_seq_set:
            negative_seqs.append(candidate)
            found = True
            break
    
    # Fallback if shuffling fails (e.g., "AAAA" cannot be shuffled to something else)
    # Generate random sequence of same length with similar GC distribution or pure random
    if not found:
        # Simple random generation fallback
        for _ in range(max_retries):
            candidate = "".join(random.choices(bases, k=len(seq)))
            if candidate not in positive_seq_set:
                negative_seqs.append(candidate)
                found = True
                break
    
    # If still not found (extremely rare), just skip or take duplication risk (we skip here)
    if not found:
        print(f"Warning: Could not generate valid negative for {seq}")
        # Last resort: random string, ignore collision risk (1 sample won't hurt)
        negative_seqs.append("".join(random.choices(bases, k=len(seq))))

# 4. Create DataFrame
pos_df = pd.DataFrame({'text': df['clean_seq'], 'labels': 1})
neg_df = pd.DataFrame({'text': negative_seqs, 'labels': 0})

combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

print(f"Total Combined Data: {len(combined_df)}")
print(combined_df['labels'].value_counts())

# 5. Split and Save
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['labels'])

train_df.to_csv(output_train, index=False)
test_df.to_csv(output_test, index=False)

print(f"Saved {len(train_df)} train samples to {output_train}")
print(f"Saved {len(test_df)} test samples to {output_test}")
