from Bio import SeqIO
import json
import os

def read_fasta(fasta_path):
    """
    Read a single sequence from a FASTA file (error if more than one).
    
    Args:
        fasta_path: Path to the FASTA file
    
    Returns:
        str: DNA sequence as string
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty, has multiple sequences, or is malformed
    """
    # Check if file exists
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    # Check if file is empty
    if os.path.getsize(fasta_path) == 0:
        raise ValueError(f"FASTA file is empty: {fasta_path}")
    
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except Exception as e:
        raise ValueError(f"Failed to parse FASTA file '{fasta_path}': {str(e)}")
    
    if len(records) == 0:
        raise ValueError(f"No valid sequences found in FASTA file: {fasta_path}")
    if len(records) > 1:
        raise ValueError(f"FASTA file contains {len(records)} sequences, but only one is allowed: {fasta_path}")
    
    record = records[0]
    return str(record.seq)


def write_output(results, output_path):
    """
    Write prediction results to a JSON file
    results: dict or list of dicts
    """
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/result.json", "w") as f:
        json.dump(results, f, indent=4)
