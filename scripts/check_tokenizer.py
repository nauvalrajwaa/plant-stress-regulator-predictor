from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("nigelhartm/PlantBERT")
    print("Tokenizer loaded successfully.")
    
    test_seq = "ACGT"
    print(f"Tokenizing {test_seq}: {tokenizer.tokenize(test_seq)}")
    
    test_seq_iupac = "ACGTRYSW"
    print(f"Tokenizing {test_seq_iupac}: {tokenizer.tokenize(test_seq_iupac)}")
    
except Exception as e:
    print(f"Error: {e}")
