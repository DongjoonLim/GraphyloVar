
import argparse
import pickle
import numpy as np
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess MAF alignments to serialized NumPy/Pickle format.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome number (e.g., '21').")
    parser.add_argument("--maf_path", type=str, default="../research/data/chr{chrom}.anc.maf", help="Path to MAF file (use {chrom} placeholder).")
    parser.add_argument("--output", type=str, default="seqDictPad_chr{chrom}.pkl", help="Output file path (use {chrom} placeholder).")
    return parser.parse_args()

def ungap(anc: str, des: str) -> tuple:
    """Remove positions where both ancestor and descendant are gaps."""
    a, d = '', ''
    for anc_char, des_char in zip(anc, des):
        if anc_char == '-' and des_char == '-':
            continue
        a += anc_char
        d += des_char
    return a, d

def get_align(input_list: list, anc_seq: list) -> dict:
    """Build aligned sequences dictionary from MAF blocks."""
    seq_dict = {a: ['N'] * input_list[0][0][1] for a in anc_seq}
    for i in tqdm(range(len(input_list))):
        for j in range(len(input_list[i])):
            item = input_list[i][j][0].split('.')[0]
            if item in seq_dict:
                seq_dict[item].extend(list(input_list[i][j][3].upper()))
        if i != 0 and (input_list[i][0][1] != input_list[i-1][0][1] + input_list[i-1][0][2]):
            for item in anc_seq:
                seq_dict[item].extend(['N'] * (input_list[i][0][1] - input_list[i-1][0][1] - input_list[i-1][0][2]))
        # Pad to max length
        lengths = [len(seq_dict[item]) for item in anc_seq]
        max_len = max(lengths)
        for item in anc_seq:
            if len(seq_dict[item]) < max_len:
                seq_dict[item].extend(['-'] * (max_len - len(seq_dict[item])))
    return seq_dict

def main():
    args = parse_arguments()
    anc_seq = [
        'hg38', 'panTro4', 'gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3', 'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1',
        'speTri2', 'jacJac1', 'micOch1', 'criGri1', 'mesAur1', 'mm10', 'rn6', 'hetGla2', 'cavPor3', 'chiLan1', 'octDeg1',
        'oryCun2', 'ochPri3', 'susScr3', 'vicPac2', 'camFer1', 'turTru2', 'orcOrc1', 'panHod1', 'bosTau8', 'oviAri3', 'capHir1',
        'equCab2', 'cerSim1', 'felCat8', 'canFam3', 'musFur1', 'ailMel1', 'odoRosDiv1', 'lepWed1', 'pteAle1', 'pteVam1', 'eptFus1',
        'myoDav1', 'myoLuc2', 'eriEur2', 'sorAra2', 'conCri1', 'loxAfr3', 'eleEdw1', 'triMan1', 'chrAsi1', 'echTel2', 'oryAfe1', 'dasNov3',
        '_HP', '_HPG', '_HPGP', '_HPGPN', '_RM', '_RMP', '_RMPC', '_HPGPNRMPC', '_CS', '_HPGPNRMPCCS', '_HPGPNRMPCCSO',
        '_CM', '_MR', '_MCM', '_MCMMR', '_JMCMMR', '_SJMCMMR', '_CO', '_CCO', '_HCCO', '_SJMCMMRHCCO', '_OO', '_SJMCMMRHCCOOO', '_HPGPNRMPCCSOTSJMCMMRHCCOOO',
        '_VC', '_TO', '_OC', '_BOC', '_PBOC', '_TOPBOC', '_VCTOPBOC', '_SVCTOPBOC',
        '_EC', '_OL', '_AOL', '_MAOL', '_CMAOL', '_FCMAOL', '_ECFCMAOL',
        '_PP', '_MM', '_EMM', '_PPEMM', '_ECFCMAOLPPEMM', '_SVCTOPBOCECFCMAOLPPEMM',
        '_SC', '_ESC', '_SVCTOPBOCECFCMAOLPPEMMESC', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC',
        '_LE', '_LET', '_CE', '_LETCE', '_LETCEO', '_LETCEOD', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD'
    ]
    print(f"Number of ancestor sequences: {len(anc_seq)}")

    maf_path = args.maf_path.format(chrom=args.chrom)
    with open(maf_path, "rb") as file:
        lines = file.readlines()

    seq_list = []
    temp_list = []
    for line in lines:
        line = str(line, 'utf-8').strip()
        if not line:
            if temp_list:
                seq_list.append(temp_list)
                temp_list = []
        elif line.startswith("s"):
            parts = line.split()
            try:
                temp_list.append([parts[1], int(parts[2]), int(parts[3]), parts[6]])
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue
    if temp_list:
        seq_list.append(temp_list)

    seq_dict_raw = get_align(seq_list, anc_seq)
    print(f"Human sequence length: {len(seq_dict_raw['hg38'])}")

    # Ungap human positions
    indices = [i for i in range(len(seq_dict_raw['hg38'])) if seq_dict_raw['hg38'][i] != '-']
    for key in tqdm(seq_dict_raw.keys()):
        seq_dict_raw[key] = [seq_dict_raw[key][i] for i in indices]
    print(f"Ungapped length: {len(indices)}")

    output_path = args.output.format(chrom=args.chrom)
    with open(output_path, 'wb') as handle:
        pickle.dump(seq_dict_raw, handle)
    np.save(output_path.replace('.pkl', '.npy'), seq_dict_raw)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
