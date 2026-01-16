import argparse
import numpy as np
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Augment data with reverse complements.")
    parser.add_argument("--tf", type=str, required=True, help="Transcription factor (e.g., 'CTCF').")
    parser.add_argument("--celltype", type=str, required=True, help="Cell type (e.g., 'K562').")
    parser.add_argument("--data_dir", type=str, default="graphs/{tf}", help="Data directory (use {tf} placeholder).")
    parser.add_argument("--length", type=int, default=1001, help="Sequence length.")
    return parser.parse_args()

def reverse_complement(input_val: int) -> int:
    """Map nucleotide to its reverse complement (int encoding)."""
    mapping = {1: 5, 5: 1, 2: 3, 3: 2}
    return mapping.get(int(input_val), input_val)

def main():
    args = parse_arguments()
    data_dir = args.data_dir.format(tf=args.tf)
    
    chromosomes_train = [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    
    X_train = np.load(f'{data_dir}/dataset_{args.length}_chr2_{args.tf}_{args.celltype}_X_train.npy').astype(np.uint8)
    y_train = np.load(f'{data_dir}/dataset_{args.length}_chr2_{args.tf}_{args.celltype}_y_train.npy').astype(np.uint8)
    
    for chrom in tqdm(chromosomes_train):
        X_train = np.concatenate((X_train, np.load(f'{data_dir}/dataset_{args.length}_chr{chrom}_{args.tf}_{args.celltype}_X_train.npy').astype(np.uint8)), axis=0)
        y_train = np.concatenate((y_train, np.load(f'{data_dir}/dataset_{args.length}_chr{chrom}_{args.tf}_{args.celltype}_y_train.npy').astype(np.uint8)), axis=0)
    
    myfunc_vec = np.vectorize(reverse_complement)
    result = myfunc_vec(X_train)
    result = np.flip(result, axis=2)
    
    X_train_revComp = np.concatenate((X_train, result), axis=2).astype(np.uint8)
    
    print(f"{args.tf} {args.celltype}")
    print(f"X shape: {X_train_revComp.shape}, y shape: {y_train.shape}")
    
    np.save(f'{data_dir}/X_revCompConcatenatedTrue{args.length}_{args.celltype}.npy', X_train_revComp)
    np.save(f'{data_dir}/y_revCompConcatenatedTrue{args.length}_{args.celltype}.npy', y_train)
    np.save(f'{data_dir}/y_revCompConcatenated{args.length}_{args.celltype}.npy', y_train)
    print("Saved augmented files.")

if __name__ == "__main__":
    main()
