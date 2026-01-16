import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess BED regions to graph inputs.")
    parser.add_argument("--bed", type=str, required=True, help="BED file path.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome (e.g., '20').")
    parser.add_argument("--output_x", type=str, required=True, help="Output X .npy path.")
    parser.add_argument("--output_y", type=str, required=True, help="Output y .npy path.")
    parser.add_argument("--context", type=int, default=100, help="Context window size.")
    parser.add_argument("--pkl_dir", type=str, default="/home/mcb/users/dlim63/conservation/data/", help="Directory with .pkl alignments.")
    return parser.parse_args()

def reverse_complement(dna: list) -> list:
    """Compute reverse complement of DNA bases."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '-': '-'}
    return [complement[base] for base in reversed(dna)]

def main():
    args = parse_arguments()
    le = LabelEncoder().fit(['A', 'C', 'G', 'T', 'N', '-'])
    
    alignment = pd.read_pickle(f'{args.pkl_dir}/seqDictPad_chr{args.chrom}.pkl')
    input_df = pd.read_csv(args.bed, delimiter=r"\s+")
    input_df = input_df[input_df.iloc[:, 0] == f'chr{args.chrom}']
    indices = input_df.iloc[:, 1]
    y_true = input_df.iloc[:, 3]
    
    species_names = [
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
    
    examples, targets = [], []
    for i, target in tqdm(zip(indices, y_true)):
        i = int(i)
        try:
            if alignment['hg38'][i] == 'N':
                continue
            example = []
            for key in species_names:
                sequence_raw = alignment[key]
                segment = sequence_raw[i - args.context : i + args.context + 1]
                sequence = le.transform(segment + reverse_complement(segment))
                example.append(sequence)
            example = np.array(example).astype('uint8')
            assert example.shape == (115, args.context * 4 + 2)
            examples.append(example)
            targets.append(target)
        except Exception as e:
            print(f"Skipping position {i}: {e}")
    
    examples = np.array(examples)
    targets = np.array(targets)
    print(f"Shapes: examples {examples.shape}, targets {targets.shape}")
    
    np.save(args.output_x, examples)
    np.save(args.output_y, targets)
    print("Saved graph inputs.")

if __name__ == "__main__":
    main()
