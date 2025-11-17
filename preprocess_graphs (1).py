import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
le = LabelEncoder()
le.fit(['A', 'C', 'G', 'T', 'N', '-'])

input_path = sys.argv[1]
chrom = sys.argv[2]
output_X = sys.argv[3]
output_y = sys.argv[4]
context = 100

alignment = pd.read_pickle(f'/home/mcb/users/dlim63/conservation/data/seqDictPad_chr{chrom}.pkl')
input = pd.read_csv(input_path, delimiter = r"\s+")
input = input.loc[input.iloc[:,0]==f'chr{chrom}']
indices = input.iloc[:,1]
y_true = input.iloc[:,3]
print(indices, y_true)

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '-': '-'}
    return [complement[base] for base in reversed(dna)]

names = ['hg38', 'panTro4','gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3', 'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1', 
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
# Fill the array with the sequences
examples = []
targets = []
for i, target in tqdm(zip(indices, y_true)):
    i = int(i)
    try:
        example = []
        if alignment['hg38'][i] =='N':
            print('contains N')
            continue
        for key in names:
            sequence_raw = alignment[key]
            sequence = le.transform(sequence_raw[i-context: i+context+1] + reverse_complement(sequence_raw[i-context: i+context+1]))
            example.append(sequence)
        example = np.array(example).astype('uint8')
        assert example.shape == (115, context*4+2)
        examples.append(example)
        targets.append(target)
    except Exception as e:
        print(e)
        continue
examples = np.array(examples)
targets =np.array(targets)
print(examples.shape, targets.shape)
np.save(f'{output_X}', examples)
np.save(f'{output_y}', targets)
