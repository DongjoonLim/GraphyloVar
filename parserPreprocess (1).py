import numpy as np
from tqdm import tqdm
import pickle
import sys

chromosome = sys.argv[1]
ancSeq = ['hg38','panTro4', 'gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3', 'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1', 
         'speTri2', 'jacJac1', 'micOch1', 'criGri1', 'mesAur1', 'mm10', 'rn6', 'hetGla2', 'cavPor3','chiLan1', 'octDeg1',
         'oryCun2', 'ochPri3',
          'susScr3',
          'vicPac2',
           'camFer1',
          'turTru2',
           'orcOrc1',
          'panHod1',
          'bosTau8',
          'oviAri3',
          'capHir1',
         'equCab2',
           'cerSim1',
          'felCat8',
          'canFam3',
          'musFur1',
          'ailMel1','odoRosDiv1','lepWed1','pteAle1','pteVam1','eptFus1','myoDav1','myoLuc2','eriEur2','sorAra2','conCri1', 'loxAfr3','eleEdw1','triMan1','chrAsi1','echTel2','oryAfe1','dasNov3',
          '_HP', '_HPG', '_HPGP', '_HPGPN', '_RM', '_RMP', '_RMPC', '_HPGPNRMPC', '_CS', '_HPGPNRMPCCS', '_HPGPNRMPCCSO' , '_HPGPNRMPCCSOT',
         '_CM', '_MR', '_MCM', '_MCMMR', '_JMCMMR', '_SJMCMMR', '_CO', '_CCO', '_HCCO', '_SJMCMMRHCCO', '_OO', '_SJMCMMRHCCOOO', '_HPGPNRMPCCSOTSJMCMMRHCCOOO'
        , '_VC', '_TO', '_OC', '_BOC', '_PBOC', '_TOPBOC', '_VCTOPBOC', '_SVCTOPBOC',
          '_EC', '_OL', '_AOL', '_MAOL', '_CMAOL' , '_FCMAOL', '_ECFCMAOL',
          '_PP', '_MM', '_EMM', '_PPEMM', '_ECFCMAOLPPEMM', '_SVCTOPBOCECFCMAOLPPEMM',
          '_SC', '_ESC', '_SVCTOPBOCECFCMAOLPPEMMESC', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC',
          '_LE', '_LET', '_CE', '_LETCE', '_LETCEO', '_LETCEOD', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD'
         ]
print(len(ancSeq))
# ancSeq = ['hg38.chr2', 'panTro4.chr2A', 'gorGor3.chr2A',  '_HP', '_HPG']

# desSeq = ['hg38', 'panTro4', 'gorGor3', 'ponAbe2', '_HP', '_HPG', '_HPGP' ]

def ungap(anc, des):
    a = ''
    d = ''
    for i in range(len(anc)):
        if anc[i] == '-' and des[i] =='-':
            continue
        else:
            a = a+ anc[i]
            d = d + des[i]
            
    return a, d

            
def getAlign(inputList, ancSeq):
    nucSet = set(['A','C','G','T','-'])
    seqDict = {}
    seqTemp = {}
    for a in tqdm(ancSeq):
        seqDict[a] = ['N']* inputList[0][0][1]
    full = []
    idList = []
    
    for i in tqdm(range(len(inputList))):
        idList = []
        lengths = []
        
    for i in tqdm(range(len(inputList))):            
        for j in range(len(inputList[i])):
            item = str(inputList[i][j][0]).split('.')[0]
            seqDict[item].extend(list(inputList[i][j][3].upper()))
        for item in ancSeq:
            if i!=0 and (inputList[i][0][1] != inputList[i-1][0][1] + inputList[i-1][0][2]):
#                 print(inputList[i][0][1], inputList[i-1][0][1], inputList[i-1][0][2])
                seqDict[item].extend( ['N'] * (inputList[i][0][1]- inputList[i-1][0][1] - inputList[i-1][0][2]))
        lengths = []
        for item in ancSeq:
#             print(seqDict[item][0])
            lengths.append(len(seqDict[item]))
        maximum = max(lengths)
        minimum = min(lengths)
        if maximum == minimum :
            continue
        else:
            for item in ancSeq:
                if len(seqDict[item]) == maximum:
#                     print(maximum, len(seqDict[item]))
                    continue
                else :
                    seqDict[item].extend(['-']*(maximum-len(seqDict[item])))
#                     print(len(seqDict[item]))

    print(len(seqDict['hg38']))
    return seqDict

# file1 = open("data/chr{}.anc.maf".format(chromosome), "rb")
file1 = open("../research/data/chr{}.anc.maf".format(chromosome), "rb")
Lines = file1.readlines()
count = 0
seqList = []
tempList = []
for line in Lines:
    line=str(line,'utf-8')
#     print(line.split())
#     print("Line{}: {}".format(count, line.strip()))
#     print(len(line.split()), line.split()[0])
    if len(line.split()) == 0 :
        if len(tempList) != 0 :
            seqList.append(tempList)
            tempList = []
    elif line.split()[0] == "s":
#         print(line.split())
        try:
            tempList.append([line.split()[1], int(line.split()[2]), int(line.split()[3]), line.split()[6]])
        except:
            print(line.split())
            continue
print(seqList[:1000])  
file1.close()
seqDictRaw = getAlign(seqList, ancSeq)
indicies = []
for i in tqdm(range(len(seqDictRaw['hg38']))):
        if seqDictRaw['hg38'][i] != '-':
            indicies.append(i)
for key in tqdm(seqDictRaw.keys()):
    temp = [seqDictRaw[key][i] for i in indicies]
    # temp = ''.join(temp)
    seqDictRaw[key] = temp
#         temp = ''
#         for index in indicies:
#             temp = temp + seqDictRaw[key][index]
#         seqDictRaw[key] = temp
print(len(indicies))
with open('seqDictPad_chr{}.pkl'.format(chromosome), 'wb') as handle:
    pickle.dump(seqDictRaw, handle)

np.save('seqDictPad_chr{}.npy'.format(chromosome), seqDictRaw)