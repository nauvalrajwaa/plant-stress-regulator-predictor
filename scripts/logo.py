import csv
import argparse
from pathlib import Path
import pandas as pd

IUPAC_CHARS = set(list("ACGTURYSWKMBDHVNacgturyswkmbdhvn"))

def check_sequence(seq: str):
    if seq is None:
        return False, 'missing'
    s = str(seq).upper().strip()
    if s == '':
        return False, 'empty'
    bad = [c for c in s if c not in IUPAC_CHARS]
    if bad:
        return False, f'illegal_chars:{"".join(sorted(set(bad)))}'
    return True, 'ok'

def check_sequences(csv_path, out_path=None, id_col='ID', seq_col='Sequence'):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(p)

    # ensure columns exist
    if seq_col not in df.columns:
        raise KeyError(f"Sequence column '{seq_col}' not found in {csv_path}")

    results = []
    for _, row in df.iterrows():
        seq = row.get(seq_col, '')
        valid, reason = check_sequence(seq)
        results.append({'ID': row.get(id_col, ''), 'Sequence': seq, 'Valid': valid, 'Reason': reason})

    df_out = pd.DataFrame(results)

    if out_path:
        df_out.to_csv(out_path, index=False)
    return df_out

def _cli():
    ap = argparse.ArgumentParser(description='Check sequences in CSV (IUPAC-aware)')
    ap.add_argument('--csv', '-c', required=True, help='Input CSV path with ID and Sequence columns')
    ap.add_argument('--out', '-o', required=False, help='Output CSV path (checked)')
    ap.add_argument('--id-col', default='ID')
    ap.add_argument('--seq-col', default='Sequence')
    args = ap.parse_args()

    out = args.out or (Path(args.csv).with_name(Path(args.csv).stem + '_checked.csv'))
    df_out = check_sequences(args.csv, out, id_col=args.id_col, seq_col=args.seq_col)
    print(f'Checked {len(df_out)} sequences, saved to {out}')

if __name__ == '__main__':
    _cli()

import argparse
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import os
import sys

def create_peptide_logo(input_file, out_dir, name, col_name='sequence', clean_noise=True, seq_type='protein'):
    # 1. Cek file input
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' tidak ditemukan.")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error baca CSV: {e}")
        sys.exit(1)

    # allow case-insensitive column name matching for sequence column
    col_candidates = [c for c in df.columns if c.lower() == col_name.lower()]
    if not col_candidates:
        print(f"Error: Kolom '{col_name}' tidak ditemukan.")
        sys.exit(1)
    col_name_actual = col_candidates[0]

    # 2. Ambil sequence
    sequences = df[col_name_actual].dropna().astype(str).tolist()
    if not sequences:
        print("Error: Data kosong.")
        sys.exit(1)

    # Filter panjang sequence agar seragam
    lengths = [len(s) for s in sequences]
    most_common_len = max(set(lengths), key=lengths.count)
    sequences = [s for s in sequences if len(s) == most_common_len]
    print(f"Memproses {len(sequences)} sequence (panjang: {most_common_len})...")

    # 3. Buat Matrix
    try:
        # Choose settings based on sequence type
        if seq_type.lower() in ('dna', 'nucleotide'):
            chars_to_ignore = '.-X*'
            color_scheme = 'classic'
        else:
            chars_to_ignore = '.-X*'
            color_scheme = 'skylign_protein'

        # matrix information bits
        ww_counts = logomaker.alignment_to_matrix(sequences=sequences, to_type='information', characters_to_ignore=chars_to_ignore)
        
        # --- CLEAN NOISE ---
        if clean_noise:
            ww_counts[ww_counts < 0.10] = 0.0
            
    except Exception as e:
        print(f"Error matriks: {e}")
        sys.exit(1)

    # 4. Plotting
    fig_width = max(8, len(sequences[0]) * 0.8) 
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    logo = logomaker.Logo(ww_counts,
                          color_scheme=color_scheme,
                          vpad=.05,
                          width=.9,
                          ax=ax)

    # Styling
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    
    ax.set_ylabel("Bits", fontsize=14)
    ax.set_xlabel("Position", fontsize=14)
    
    # --- PERUBAHAN DI SINI ---
    # Baris title di bawah ini sudah dihapus/dikomentari
    # ax.set_title(f"Sequence Logo: {name}", fontsize=16) 
    
    # Fix Skala Y (4.5 bits)
    ax.set_ylim([0, 4.5])
    ax.set_xticks(range(len(ww_counts)))

    # 5. Simpan
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Sukses! Plot bersih tersimpan di: {output_path}")
    return output_path


def create_logo_from_list(sequences, out_dir, name, clean_noise=True, seq_type='protein'):
    """Create and save a logo from an in-memory list of sequences.
    This function mirrors the behaviour of create_peptide_logo but
    avoids reading/writing intermediate CSV files.
    """
    if not sequences:
        raise ValueError('Empty sequences list')

    # ensure strings
    sequences = [str(s) for s in sequences if s is not None]

    # Filter length to most common length
    lengths = [len(s) for s in sequences]
    most_common_len = max(set(lengths), key=lengths.count)
    sequences = [s for s in sequences if len(s) == most_common_len]

    # Choose settings based on sequence type
    if seq_type.lower() in ('dna', 'nucleotide'):
        chars_to_ignore = '.-X*'
        color_scheme = 'classic'
    else:
        chars_to_ignore = '.-X*'
        color_scheme = 'skylign_protein'

    try:
        ww_counts = logomaker.alignment_to_matrix(sequences=sequences, to_type='information', characters_to_ignore=chars_to_ignore)
        if clean_noise:
            ww_counts[ww_counts < 0.10] = 0.0
    except Exception as e:
        raise RuntimeError(f"Error creating matrix: {e}")

    fig_width = max(8, len(sequences[0]) * 0.8)
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
    ax.set_ylim([0, 4.5])
    ax.set_xticks(range(len(ww_counts)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('--out-dir', type=str, default='plot')
    parser.add_argument('--name', type=str, default='seq_logo')
    parser.add_argument('--col', type=str, default='sequence')
    parser.add_argument('--keep-noise', action='store_true', help='Jangan hapus huruf-huruf kecil (noise)')

    args = parser.parse_args()
    do_clean = not args.keep_noise

    create_peptide_logo(args.input_file, args.out_dir, args.name, args.col, do_clean)