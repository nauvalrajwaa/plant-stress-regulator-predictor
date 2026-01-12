import pandas as pd

file_path = "/Users/user/Downloads/02. PROJECTS/Stress-region-predictor/train/data/merged_place_seq.csv"

try:
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Length Analysis
df['Length'] = df['Sequence'].astype(str).str.strip().apply(len)

print("\nSequence Length Statistics:")
print(df['Length'].describe())

print("\nTop 5 shortest sequences:")
print(df.nsmallest(5, 'Length')[['ID', 'Sequence', 'Length']])

print("\nTop 5 longest sequences:")
print(df.nlargest(5, 'Length')[['ID', 'Sequence', 'Length']])

# Class Analysis
num_classes = df['ID'].nunique()
print(f"\nNumber of unique IDs (Classes): {num_classes}")

print("\nTop 20 most frequent classes:")
print(df['ID'].value_counts().head(20))

# Check for singletons
class_counts = df['ID'].value_counts()
singletons = class_counts[class_counts == 1]
print(f"\nNumber of classes with only 1 sample: {len(singletons)}")
print(f"Number of classes with < 5 samples: {len(class_counts[class_counts < 5])}")
