
import argparse
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from Bio.Seq import Seq

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SPECIES_ORDER = [
    'hg38', 'panTro4','gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3', 'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1', 
    'speTri2', 'jacJac1', 'micOch1', 'criGri1', 'mesAur1', 'mm10', 'rn6', 'hetGla2', 'cavPor3','chiLan1', 'octDeg1',
    'oryCun2', 'ochPri3','susScr3','vicPac2','camFer1','turTru2', 'orcOrc1', 'panHod1','bosTau8','oviAri3','capHir1','equCab2','cerSim1','felCat8','canFam3',
    'musFur1','ailMel1', 'odoRosDiv1', 'lepWed1','pteAle1','pteVam1',  'eptFus1', 'myoDav1','myoLuc2','eriEur2',
    'sorAra2', 'conCri1','loxAfr3', 'eleEdw1','triMan1','chrAsi1','echTel2','oryAfe1','dasNov3',
    '_HP', '_HPG', '_HPGP', '_HPGPN', '_RM', '_RMP', '_RMPC', '_HPGPNRMPC', '_CS', '_HPGPNRMPCCS', '_HPGPNRMPCCSO' , '_HPGPNRMPCCSOT',
    '_CM', '_MR', '_MCM', '_MCMMR', '_JMCMMR', '_SJMCMMR', '_CO', '_CCO', '_HCCO', '_SJMCMMRHCCO', '_OO', '_SJMCMMRHCCOOO', '_HPGPNRMPCCSOTSJMCMMRHCCOOO'
    , '_VC', '_TO', '_OC', '_BOC', '_PBOC', '_TOPBOC', '_VCTOPBOC', '_SVCTOPBOC',
    '_EC', '_OL', '_AOL', '_MAOL', '_CMAOL' , '_FCMAOL', '_ECFCMAOL',
    '_PP', '_MM', '_EMM', '_PPEMM', '_ECFCMAOLPPEMM', '_SVCTOPBOCECFCMAOLPPEMM',
    '_SC', '_ESC', '_SVCTOPBOCECFCMAOLPPEMMESC', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC',
    '_LE', '_LET', '_CE', '_LETCE', '_LETCEO', '_LETCEOD', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD'
]

def reverse_complement(seq_str):
    """Returns the reverse complement of a DNA string."""
    # Handle gaps/N manually if needed, or rely on BioPython
    # Using simple dictionary for speed if strings are clean, but BioPython is safer
    table = str.maketrans("ACGTN-", "TGCAN-")
    return seq_str.translate(table)[::-1]

def one_hot_encode_targets(seq_str):
    """Encodes the target human base for the output (5 classes: A, C, G, T, -)."""
    mapping = {'A':0, 'C':1, 'G':2, 'T':3, '-':4, 'N':4} # Treat N as gap for target or ignore
    vec = np.zeros(5)
    idx = mapping.get(seq_str, 4)
    if idx < 5:
        vec[idx] = 1
    return vec

def preprocess_chromosome(chrom, input_dir, output_dir, flank_size=100, num_mutated=100000, num_conserved=400000):
    pkl_path = os.path.join(input_dir, f'seqDictPad_chr{chrom}.pkl')
    print(f"Loading {pkl_path}...")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find {pkl_path}")
        
    alignment = pd.read_pickle(pkl_path)
    
    human = alignment['hg38']
    ancestor = alignment['_HP'] # Human-Chimp Ancestor
    
    # 1. Identify Mutations vs Conserved Sites
    print("Identifying mutation sites...")
    seq_len = len(human)
    diff_indices = [i for i in range(seq_len) if human[i] != ancestor[i] and human[i] != 'N']
    conserved_indices = list(set(range(seq_len)) - set(diff_indices))
    
    # 2. Sampling
    print(f"Sampling {num_mutated} mutated and {num_conserved} conserved sites...")
    # Handle cases where we have fewer sites than requested
    mutated_sampled = random.sample(diff_indices, min(len(diff_indices), num_mutated))
    conserved_sampled = random.sample(conserved_indices, min(len(conserved_indices), num_conserved))
    
    all_indices = mutated_sampled + conserved_sampled
    random.shuffle(all_indices)
    
    # 3. Label Encoder setup (Inputs are integers 0-5)
    le = LabelEncoder()
    le.fit(['A', 'C', 'G', 'T', 'N', '-'])
    
    X = []
    Y = []
    
    print("Extracting windows and encoding...")
    for i in tqdm(all_indices):
        # Boundary checks
        if i < flank_size or i >= seq_len - flank_size:
            continue
            
        # Target Preparation
        # Y is (5 classes for identity) + (1 class for binary polymorphism/mutation status)
        # Here we define "mutation status" as derived != ancestor
        current_base = human[i]
        is_mutated = 1 if i in diff_indices else 0
        
        target_vec = one_hot_encode_targets(current_base)
        y_sample = np.append(target_vec, is_mutated) # Size 6
        
        # Input Preparation
        # Shape: (115 species, 402 bp) -> 402 comes from (100 flank + 1 center + 100 flank) * 2 for RevComp
        example_matrix = []
        try:
            for species in SPECIES_ORDER:
                seq_raw = alignment[species]
                # Extract Window
                segment = seq_raw[i - flank_size : i + flank_size + 1]
                # Reverse Complement
                segment_rc = reverse_complement(segment)
                # Combine
                combined_seq = list(segment + segment_rc)
                
                # Integer Encode
                encoded_seq = le.transform(combined_seq)
                example_matrix.append(encoded_seq)
            
            X.append(np.array(example_matrix, dtype='uint8'))
            Y.append(y_sample)
            
        except Exception as e:
            # Skip malformed windows
            continue

    # 4. Save
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Final Shapes -> X: {X.shape}, Y: {Y.shape}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, f'X_train_chr{chrom}.npy'), X)
    np.save(os.path.join(output_dir, f'y_train_chr{chrom}.npy'), Y)
    print("Saved .npy files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome number (e.g., 21)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .pkl alignment files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")
    args = parser.parse_args()
    
    preprocess_chromosome(args.chrom, args.input_dir, args.output_dir)
