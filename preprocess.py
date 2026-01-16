import argparse
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from Bio.Seq import Seq

SPECIES_ORDER = [
    'hg38', 'panTro4', 'gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3', 'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1',
    'speTri2', 'jacJac1', 'micOch1', 'criGri1', 'mesAur1', 'mm10', 'rn6', 'hetGla2', 'cavPor3', 'chiLan1', 'octDeg1',
    'oryCun2', 'ochPri3', 'susScr3', 'vicPac2', 'camFer1', 'turTru2', 'orcOrc1', 'panHod1', 'bosTau8', 'oviAri3', 'capHir1', 'equCab2', 'cerSim1', 'felCat8', 'canFam3',
    'musFur1', 'ailMel1', 'odoRosDiv1', 'lepWed1', 'pteAle1', 'pteVam1', 'eptFus1', 'myoDav1', 'myoLuc2', 'eriEur2',
    'sorAra2', 'conCri1', 'loxAfr3', 'eleEdw1', 'triMan1', 'chrAsi1', 'echTel2', 'oryAfe1', 'dasNov3',
    '_HP', '_HPG', '_HPGP', '_HPGPN', '_RM', '_RMP', '_RMPC', '_HPGPNRMPC', '_CS', '_HPGPNRMPCCS', '_HPGPNRMPCCSO', '_HPGPNRMPCCSOT',
    '_CM', '_MR', '_MCM', '_MCMMR', '_JMCMMR', '_SJMCMMR', '_CO', '_CCO', '_HCCO', '_SJMCMMRHCCO', '_OO', '_SJMCMMRHCCOOO', '_HPGPNRMPCCSOTSJMCMMRHCCOOO',
    '_VC', '_TO', '_OC', '_BOC', '_PBOC', '_TOPBOC', '_VCTOPBOC', '_SVCTOPBOC',
    '_EC', '_OL', '_AOL', '_MAOL', '_CMAOL', '_FCMAOL', '_ECFCMAOL',
    '_PP', '_MM', '_EMM', '_PPEMM', '_ECFCMAOLPPEMM', '_SVCTOPBOCECFCMAOLPPEMM',
    '_SC', '_ESC', '_SVCTOPBOCECFCMAOLPPEMMESC', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC',
    '_LE', '_LET', '_CE', '_LETCE', '_LETCEO', '_LETCEOD', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD'
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess chromosome alignments for training.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome number (e.g., '21').")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .pkl alignments.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files.")
    parser.add_argument("--flank_size", type=int, default=100, help="Flank size around center.")
    parser.add_argument("--num_mutated", type=int, default=100000, help="Number of mutated sites to sample.")
    parser.add_argument("--num_conserved", type=int, default=400000, help="Number of conserved sites to sample.")
    return parser.parse_args()

def reverse_complement(seq_str: str) -> str:
    """Compute reverse complement of DNA sequence."""
    table = str.maketrans("ACGTN-", "TGCAN-")
    return seq_str.translate(table)[::-1]

def one_hot_encode_targets(seq_str: str) -> np.ndarray:
    """One-hot encode target base (A,C,G,T,-/N)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, 'N': 4}
    vec = np.zeros(5)
    idx = mapping.get(seq_str, 4)
    vec[idx] = 1
    return vec

def preprocess_chromosome(args):
    pkl_path = os.path.join(args.input_dir, f'seqDictPad_chr{args.chrom}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing {pkl_path}")
    
    print(f"Loading {pkl_path}...")
    alignment = pd.read_pickle(pkl_path)
    
    human = alignment['hg38']
    ancestor = alignment['_HP']  # Human-Chimp Ancestor
    
    seq_len = len(human)
    diff_indices = [i for i in range(seq_len) if human[i] != ancestor[i] and human[i] != 'N']
    conserved_indices = list(set(range(seq_len)) - set(diff_indices))
    
    print(f"Sampling {args.num_mutated} mutated and {args.num_conserved} conserved sites...")
    mutated_sampled = random.sample(diff_indices, min(len(diff_indices), args.num_mutated))
    conserved_sampled = random.sample(conserved_indices, min(len(conserved_indices), args.num_conserved))
    
    all_indices = mutated_sampled + conserved_sampled
    random.shuffle(all_indices)
    
    le = LabelEncoder().fit(['A', 'C', 'G', 'T', 'N', '-'])
    
    X, Y = [], []
    print("Extracting windows and encoding...")
    for i in tqdm(all_indices):
        if i < args.flank_size or i >= seq_len - args.flank_size:
            continue
        
        current_base = human[i]
        is_mutated = 1 if i in diff_indices else 0
        target_vec = one_hot_encode_targets(current_base)
        y_sample = np.append(target_vec, is_mutated)
        
        example_matrix = []
        try:
            for species in SPECIES_ORDER:
                seq_raw = alignment[species]
                segment = seq_raw[i - args.flank_size: i + args.flank_size + 1]
                segment_rc = reverse_complement(segment)
                combined_seq = list(segment + segment_rc)
                encoded_seq = le.transform(combined_seq)
                example_matrix.append(encoded_seq)
            X.append(np.array(example_matrix, dtype='uint8'))
            Y.append(y_sample)
        except Exception as e:
            print(f"Skipping window at {i}: {e}")
            continue
    
    X = np.array(X)
    Y = np.array(Y)
    print(f"Final shapes: X {X.shape}, Y {Y.shape}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f'X_train_chr{args.chrom}.npy'), X)
    np.save(os.path.join(args.output_dir, f'y_train_chr{args.chrom}.npy'), Y)
    print("Saved .npy files.")

if __name__ == "__main__":
    args = parse_arguments()
    preprocess_chromosome(args)
