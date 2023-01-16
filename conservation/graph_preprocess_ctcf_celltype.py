import re
import pickle
import itertools
import random
import string
from tqdm import tqdm
import pandas as pd
import numpy as np
from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader, MixedLoader
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.gcn_filter import GCNFilter
import networkx as nx
import scipy.sparse as sp
import sys

# In[2]:
#val 18, test 20, train other
celltype = sys.argv[1]
chromosomes = [20] #,4,5,6,7,8,9,10]
seqDictRaw = [] 
for i in chromosomes:
    print(i)
    seqDictRaw.append(pd.read_pickle('data/seqDictPad1000000_chr{}.pkl'.format(i)))

ccres = []
for chromosome in chromosomes :
    ccre =  pd.read_csv('data/{}.bed'.format(celltype), delimiter = '\t', header = None)
    ccre = ccre.loc[(ccre[9]=='pELS,CTCF-bound') | (ccre[9]=='dELS,CTCF-bound') | (ccre[9]=='CTCF-only,CTCF-bound')| (ccre[9]=='DNase-H3K4me3,CTCF-bound') | (ccre[9]=='PLS,CTCF-bound')  ]
    ccre = ccre.loc[ccre[0] == 'chr{}'.format(chromosome)]
    ccres.append(ccre)

# exons = []
# for chromosome in chromosomes :
#     exon =  pd.read_csv('gencode_exons.bed', delimiter = '\t', header = None)
#     exon = exon.loc[exon[0] == 'chr{}'.format(chromosome)]
#     exon = exon.loc[exon[5] == '+']
#     exons.append(exon)

def getFunctionality(ccres):
    functionalities = []
    for i in range (len(ccres)):
        starts = list(ccres[i].iloc[:,1].astype(int))
        stops = list(ccres[i].iloc[:,2].astype(int))
        length = len(seqDictRaw[i]['hg38'])
        functionality = [0]*length
        for start, stop in zip(starts, stops):
            if stop < length:
                for j in range(start, stop):
                    functionality[j] = 1
            else :
                break
        functionalities.append(functionality)
    return functionalities
# functionalities = getFunctionality(exons)
functionalities = getFunctionality(ccres)


# ccre =  pd.read_csv('ccres.bed', delimiter = '\t', header = None)
# ccre = ccre.loc[ccre[0] == 'chr1']
# starts = list(ccre.iloc[:,1].astype(int))
# stops = list(ccre.iloc[:,2].astype(int))
# functionalities = [0]*len(seqDictRaw['hg38'])
# for start, stop in zip(starts, stops):
#     if stop < len(seqDictRaw['hg38']):
#         for i in range(start, stop):
#             functionalities[i] = 1
#     else :
#         break

# exons =  pd.read_csv('gencode_exons.bed', delimiter = '\t', header = None)
# exons = exons.loc[exons[0] == 'chr1']
# exons = exons.loc[exons[5] == '+']
# print(exons.head())
# starts = list(exons.iloc[:,1].astype(int))
# stops = list(exons.iloc[:,2].astype(int))
# functionalities = [0]*len(seqDictRaw['hg38'])
# for start, stop in zip(starts, stops):
#     if stop < len(seqDictRaw['hg38']):
#         for i in range(start, stop):
#             functionalities[i] = 1
#     else :
#         break

# exons =  pd.read_csv('gencode_3utr.bed', delimiter = '\t', header = None)
# exons = exons.loc[exons[0] == 'chr{}'.format(chromosome)]
# exons = exons.loc[exons[5] == '+']
# print(exons.head())
# starts = list(exons.iloc[:,1].astype(int))
# stops = list(exons.iloc[:,2].astype(int))
# functionalities = [0]*len(seqDictRaw['hg38'])
# for start, stop in zip(starts, stops):
#     if stop < len(seqDictRaw['hg38']):
#         for i in range(start, stop):
#             functionalities[i] = 1
#     else :
#         break

# print(functionalities[170000:200000])


def mergeDict(listDict):
    dic = {}
    for key in listDict[0].keys():
        dic[key] = ''
    for i in range(len(listDict)):
        for key in listDict[0].keys():
            dic[key] = dic[key]+ listDict[i][key]
    return dic
        
        
def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

seqDictRaw = mergeDict(seqDictRaw)
functionalities = flatten(functionalities)

print(len(seqDictRaw['hg38']))
print(len(functionalities))



def oneHot(nuc):
    if nuc == 'A':
        return [1,0,0,0,0]
    elif nuc == 'C':
        return [0,1,0,0,0]
    elif nuc == 'G':
        return [0,0,1,0,0]
    elif nuc == 'T':
        return [0,0,0,1,0]
    elif nuc == '-':
        return [0,0,0,0,1]
    else :
        return [0,0,0,0,0]


def binaryTarget(val1, val2, functionality):
    # if functionality :
    #         print(val1, val2, np.array_equal(val1, val2))
    if functionality: # positive example is conserved
        return 1 # np.array([0, 1])
    elif  not functionality: #np.array([1, 0]) # negative example is mutated
        return 0
    else :
        return -1

print(len(seqDictRaw['hg38']))
print(len(seqDictRaw['panTro4']))
print(len(seqDictRaw['gorGor3']))
print(len(seqDictRaw['ponAbe2']))
print(len(seqDictRaw['_HP']))
print(len(seqDictRaw['_HPG']))
print(len(seqDictRaw['_HPGP']))
for key in seqDictRaw.keys():
    seqDictRaw[key] = list(seqDictRaw[key])
seqDictRaw['y'] = []
for i, func in tqdm(zip(range(len(seqDictRaw['hg38'])), functionalities)):
    # human = binaryTarget(seqDictRaw['hg38'][i],seqDictRaw['_HP'][i],func)
    # boolean = human
    # chimp = binaryTarget(seqDictRaw['panTro4'][i],seqDictRaw['_HP'][i],func)
    # if human == 0 or chimp ==0:
    #     boolean = 0
    # else :
    #     boolean = human or chimp
    seqDictRaw['y'].append(func)
delete_index = []
for i, a, b, c, d, e, f in tqdm(zip(range(len(seqDictRaw['hg38'])), seqDictRaw['panTro4'], seqDictRaw['gorGor3'], seqDictRaw['ponAbe2'], seqDictRaw['_HP'], seqDictRaw['_HPG'] , seqDictRaw['_HPGP'])):
    if d == 'N' or d == '-' or a == '-' or (len(set([b, c, d, e, f])) !=1):
        delete_index.append(i) 
    
rawdf = pd.DataFrame(seqDictRaw)
seqdf = rawdf.drop(delete_index)


negative = seqdf.loc[seqdf['y'] == 0]
positive = seqdf.loc[seqdf['y'] == 1]
print(negative.shape, positive.shape)
negative = negative.iloc[200:-200]
positive = positive.iloc[200:-200]
negative = negative.loc[negative['y'] == 0].sample(min(negative.shape[0], positive.shape[0]), random_state=42)
positive = positive.loc[positive['y'] == 1].sample(min(negative.shape[0], positive.shape[0]), random_state=42)
print(negative.shape)
seqdf = pd.concat([positive, negative])
# y = LabelBinarizer().fit_transform(seqdf['y'])
# seqdf['y'] = y



from sklearn.preprocessing import LabelBinarizer, LabelEncoder
le = LabelEncoder()
le.fit(['A', 'C', 'G', 'T', 'N', '-'])
print(le.transform(['A', 'C', 'G', 'T', 'N', '-']))
print(list(le.classes_))
#Initialize the graph
G = nx.Graph(name='G')

#Create nodes

#Each node is assigned node feature which corresponds to the node name
names = ['hg38', 'panTro4', 'gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3', 'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1', 
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
for a,b in enumerate(names):
    G.add_node(b, name=a)
# for i in range(5):
#     G.add_node(i, name=i)
#edges
edges = [('hg38','_HP'), ('panTro4','_HP'),('gorGor3','_HPG'),('ponAbe2','_HPGP'),('_HP','_HPG'), ('_HPG','_HPGP'), ('nomLeu3', '_HPGPN'), 
         ('_HPGP', '_HPGPN'), ('_HPGPN', '_HPGPNRMPC'), ('_HPGPNRMPC', '_HPGPNRMPCCS'), ('_HPGPNRMPCCS', '_HPGPNRMPCCSO'), 
         ('_HPGPNRMPCCSO', '_HPGPNRMPCCSOT'), ('rheMac3', '_RM'), ('macFas5', '_RM'), ('_RM', '_RMP'), ('papAnu2', '_RMP'),
         ('_RMP', '_RMPC'), ('chlSab2', '_RMPC'), ('_RMPC', '_HPGPNRMPC'), ('calJac3','_CS'), ('saiBol1','_CS') , ('_CS', '_HPGPNRMPCCS'),
         ('otoGar3', '_HPGPNRMPCCSO'), ('tupChi1','_HPGPNRMPCCSOT'), 
         ('speTri2', '_SJMCMMR'), ('_SJMCMMR','_JMCMMR'), ('jacJac1','_JMCMMR'), ('micOch1', '_MCM'), ('_MCMMR','_JMCMMR'), ('_MCM','_MCMMR'),
         ('_CM','_MCM'), ('_MR','_MCMMR'), ('criGri1', '_CM'), ('mesAur1', '_CM'), ('mm10','_MR'), ('rn6','_MR'), ('_SJMCMMRHCCO', '_HCCO'),
         ('_SJMCMMRHCCO','_SJMCMMR'),
         ('_SJMCMMRHCCO','_SJMCMMRHCCOOO'), ('_HPGPNRMPCCSOTSJMCMMRHCCOOO','_HPGPNRMPCCSOT'), ('_HPGPNRMPCCSOTSJMCMMRHCCOOO','_SJMCMMRHCCOOO'),
         ('_CCO', '_HCCO'),('_CO', '_CCO'),('_OO','_SJMCMMRHCCOOO'),('hetGla2', '_HCCO'),('cavPor3', '_CCO'),('chiLan1', '_CO'),
         ('octDeg1', '_CO'),('oryCun2', '_OO'),('ochPri3', '_OO'),
         ('vicPac2','_VC'), ('camFer1','_VC'), ('susScr3', '_SVCTOPBOC'), ('turTru2','_TO'), ('orcOrc1','_TO'),
         ('oviAri3','_OC'), ('capHir1', '_OC'), ('bosTau8','_BOC'), ('_OC','_BOC'), ('panHod1','_PBOC'),
         ('_BOC','_PBOC'), ('_PBOC','_TOPBOC') , ('_TO','_TOPBOC'), ('_TOPBOC','_VCTOPBOC'), ('_VC','_VCTOPBOC'),
         ('_VCTOPBOC','_SVCTOPBOC'), ('susScr3','_SVCTOPBOC'),
         ('equCab2','_EC'), ('cerSim1','_EC'), ('odoRosDiv1','_OL'), ('lepWed1','_OL'), ('_OL','_AOL'),
         ('ailMel1','_AOL'), ('_AOL', '_MAOL'), ('musFur1', '_MAOL'), ('_MAOL','_CMAOL'), ('canFam3','_CMAOL'),
         ('_CMAOL','_FCMAOL'), ('felCat8','_FCMAOL'), ('_FCMAOL', '_ECFCMAOL'), ('_EC', '_ECFCMAOL'),
         ('pteAle1', '_PP'), ('pteVam1','_PP'), ('myoDav1','_MM'), ('myoLuc2','_MM'), ('eptFus1','_EMM'), ('_MM','_EMM'),
         ('_EMM','_PPEMM'), ('_PP','_PPEMM'),('_PPEMM','_ECFCMAOLPPEMM'),('_ECFCMAOL','_ECFCMAOLPPEMM'),('_ECFCMAOLPPEMM','_SVCTOPBOCECFCMAOLPPEMM'),('_SVCTOPBOC','_SVCTOPBOCECFCMAOLPPEMM'),
         ('sorAra2','_SC'), ('conCri1', '_SC'), ('_SC','_ESC'), ('eriEur2','_ESC'),
         ('_ESC','_SVCTOPBOCECFCMAOLPPEMMESC'), ('_SVCTOPBOCECFCMAOLPPEMM','_SVCTOPBOCECFCMAOLPPEMMESC'), ('_SVCTOPBOCECFCMAOLPPEMM','_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC'), ('_HPGPNRMPCCSOTSJMCMMRHCCOOO','_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC'),
         ('loxAfr3','_LE'), ('eleEdw1','_LE'), ('triMan1','_LET'), ('_LE', '_LET'), ('chrAsi1','_CE'), ('echTel2','_CE'),
         ('_LET','_LETCE'), ('_CE','_LETCE'), ('_LETCE','_LETCEO'), ('oryAfe1','_LETCEO'),('_LETCEO','_LETCEOD'), ('dasNov3', '_LETCEOD'), ('_LETCEOD', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD'),('_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC', '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD')
        ]

# edges = [(0,3),(1,3),(2,4),(3,4)]
G.add_edges_from(edges)

print('Graph Info:\n', nx.info(G))

print('\nGraph Nodes: ', G.nodes.data())

#Plot the graph
# plt.figure(figsize=(15,15)) 
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
print(len(names), len(edges))

A = np.array(nx.attr_matrix(G, node_attr='name')[0])
print(A)
class ModDataset(Dataset):

    def __init__(self,seqdf, rawdf, a, margin, **kwargs):
        self.rawdf = rawdf
        self.seqdf = seqdf
        self.margin = margin
        self.a = a
        self.indices = seqdf.index
        self.n_samples = self.seqdf.shape[0]
        super().__init__(**kwargs)

    def read(self):
        def make_graph(i):
            n = self.seqdf.shape[0]

            # Node features
            idx = self.indices[i] #+ self.margin
#             print(rawdf.loc[idx])
            x = np.expand_dims(le.transform(self.rawdf.iloc[idx-self.margin][0:-1]), axis=1)
#             print(x)
            for ind in range(idx -self.margin+1, idx + self.margin +1):
                x = np.append(x, np.expand_dims(le.transform(self.rawdf.iloc[ind][0:-1]), axis=1), axis = 1) 
#                 x = x+ le.transform(self.rawdf.iloc[ind][1:-1])
            x = list(x)
#             print(idx, x)

            x = np.asarray(x).astype('float32')
#             print(x.shape)
            
            y = self.rawdf['y'].loc[idx]
            mat = np.zeros((2))
            mat[y] = 1
            y = mat

            # Edges
            I = np.identity(A.shape[0])
            AI = A #+I
            a = sp.csr_matrix(AI)

            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph(i) for i in tqdm(range(100, self.n_samples-100))]



def toPickle(name, data):
    filehandler = open(name, 'wb') 
    pickle.dump(data, filehandler)

def preprocess(dataset_15):
    X_train = []
    y_train = []
    for i in range(dataset_15.n_graphs):
        X_train.append(dataset_15[i].x)
        y_train.append(dataset_15[i].y)
    X_train = np.array(X_train)#.reshape(-1, 113, 15, 1)
    y_train = np.array(y_train)
    return X_train, y_train

# toPickle('graphs/train_labels_chr{}_exon.pkl'.format('All'), list(seqdf['y']))
# toPickle('graphs/train_indices_chr{}_exon.pkl'.format('All'), list(seqdf.index))
# dataset_15 = ModDataset(seqdf, rawdf, A, 7, transform =NormalizeAdj())
# toPickle('graphs/dataset_15_chr{}_exon.pkl'.format('All'), dataset_15)

# dataset_51 = ModDataset(seqdf, rawdf, A, 25, transform =NormalizeAdj())
# toPickle('graphs/dataset_51_chr{}_exon.pkl'.format('All'), dataset_51)

# dataset_101 = ModDataset(seqdf, rawdf, A, 50, transform =NormalizeAdj())
# toPickle('graphs/dataset_101_chr{}_exon.pkl'.format('All'), dataset_101)

# dataset_201 = ModDataset(seqdf, rawdf, A, 100, transform =NormalizeAdj())
# toPickle('graphs/dataset_201_chr{}_exon.pkl'.format('All'), dataset_201)


# toPickle('graphs/train_labels_chr{}_3utr.pkl'.format(chromosome), list(seqdf['y']))
# toPickle('graphs/train_indices_chr{}_3utr.pkl'.format(chromosome), list(seqdf.index))
# dataset_15 = ModDataset(seqdf, rawdf, A, 7, transform =NormalizeAdj())
# toPickle('graphs/dataset_15_chr{}_3utr.pkl'.format(chromosome), dataset_15)

# dataset_51 = ModDataset(seqdf, rawdf, A, 25, transform =NormalizeAdj())
# toPickle('graphs/dataset_51_chr{}_3utr.pkl'.format(chromosome), dataset_51)

# dataset_101 = ModDataset(seqdf, rawdf, A, 50, transform =NormalizeAdj())
# toPickle('graphs/dataset_101_chr{}_3utr.pkl'.format(chromosome), dataset_101)

# dataset_201 = ModDataset(seqdf, rawdf, A, 100, transform =NormalizeAdj())
# toPickle('graphs/dataset_201_chr{}_3utr.pkl'.format('All'), dataset_201)


# toPickle('graphs/train_labels_chr{}_ccre_nonccre_{}.pkl'.format('All', celltype), list(seqdf['y']))
# toPickle('graphs/train_indices_chr{}_ccre_nonccre_{}.pkl'.format('All', celltype), list(seqdf.index))

dataset_101 = ModDataset(seqdf, rawdf, A, 50, transform =NormalizeAdj())
# toPickle('graphs/dataset_101_chr{}_ccre_nonccre.pkl'.format('All'), dataset_101)
X_train101, y_train101 = preprocess(dataset_101)
# np.save('graphs/dataset_101_chr{}_ctcf_{}_X_train.npy'.format('All', celltype), X_train101)
# np.save('graphs/dataset_101_chr{}_ctcf_{}_y_train.npy'.format('All', celltype), y_train101)
np.save('graphs/dataset_101_chr{}_ctcf_{}_X_test.npy'.format('All', celltype), X_train101)
np.save('graphs/dataset_101_chr{}_ctcf_{}_y_test.npy'.format('All', celltype), y_train101)
# np.save('graphs/dataset_101_chr{}_ctcf_{}_X_val.npy'.format('All', celltype), X_train101)
# np.save('graphs/dataset_101_chr{}_ctcf_{}_y_val.npy'.format('All', celltype), y_train101)

# dataset_51 = ModDataset(seqdf, rawdf, A, 25, transform =NormalizeAdj())
# # toPickle('graphs/dataset_101_chr{}_ccre_nonccre.pkl'.format('All'), dataset_101)
# X_train51, y_train51 = preprocess(dataset_51)
# # np.save('graphs/dataset_101_chr{}_ccre_nonccre_{}_X_train.npy'.format('All', celltype), X_train101)
# # np.save('graphs/dataset_101_chr{}_ccre_nonccre_{}_y_train.npy'.format('All', celltype), y_train101)
# np.save('graphs/dataset_101_chr{}_ccre_nonccre_{}_X_train.npy'.format('All', celltype), X_train101)
# np.save('graphs/dataset_101_chr{}_ccre_nonccre_{}_y_train.npy'.format('All', celltype), y_train101)
# # np.save('graphs/dataset_51_chr{}_ccre_nonccre_{}_X_val.npy'.format('All', celltype), X_train51)
# # np.save('graphs/dataset_51_chr{}_ccre_nonccre_{}_y_val.npy'.format('All', celltype), y_train51)

# dataset_201 = ModDataset(seqdf, rawdf, A, 100, transform =NormalizeAdj())
# # toPickle('graphs/dataset_201_chr{}_ccre_nonccre.pkl'.format('All'), dataset_201)
# X_train101, y_train101 = preprocess(dataset_101)
# np.save('graphs/dataset_201_chr{}_ccre_nonccre_{}_X_train.npy'.format('All', celltype), X_train201)
# np.save('graphs/dataset_201_chr{}_ccre_nonccre_{}_y_train.npy'.format('All', celltype), y_train201)

# dataset_15 = ModDataset(seqdf, rawdf, A, 7, transform =NormalizeAdj())
# # toPickle('graphs/dataset_15_chr{}_ccre_nonccre.pkl'.format('All'), dataset_15)
# X_train15, y_train15 = preprocess(dataset_15)
# np.save('graphs/dataset_15_chr{}_ccre_nonccre_{}_X_train.npy'.format('All', celltype), X_train15)
# np.save('graphs/dataset_15_chr{}_ccre_nonccre_{}_y_train.npy'.format('All', celltype), y_train15)

# dataset_51 = ModDataset(seqdf, rawdf, A, 25, transform =NormalizeAdj())
# # toPickle('graphs/dataset_51_chr{}_ccre_nonccre.pkl'.format('All'), dataset_51)
# X_train51, y_train51 = preprocess(dataset_51)
# # np.save('graphs/dataset_51_chr{}_ccre_nonccre_{}_X_train.npy'.format('All', celltype), X_train51)
# # np.save('graphs/dataset_51_chr{}_ccre_nonccre_{}_y_train.npy'.format('All', celltype), y_train51)
# np.save('graphs/dataset_51_chr{}_ccre_nonccre_{}_X_test.npy'.format('All', celltype), X_train51)
# np.save('graphs/dataset_51_chr{}_ccre_nonccre_{}_y_test.npy'.format('All', celltype), y_train51)





# toPickle('graphs/train_labels_chr{}.pkl'.format(chromosome), list(seqdf['y']))
# toPickle('graphs/train_indices_chr{}.pkl'.format(chromosome), list(seqdf.index))
# dataset_15 = ModDataset(seqdf, rawdf, A, 7, transform =NormalizeAdj())
# toPickle('graphs/dataset_15_chr{}.pkl'.format(chromosome), dataset_15)

# dataset_51 = ModDataset(seqdf, rawdf, A, 25, transform =NormalizeAdj())
# toPickle('graphs/dataset_51_chr{}.pkl'.format(chromosome), dataset_51)

# dataset_101 = ModDataset(seqdf, rawdf, A, 50, transform =NormalizeAdj())
# toPickle('graphs/dataset_101_chr{}.pkl'.format(chromosome), dataset_101)