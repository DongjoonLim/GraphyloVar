from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
# import keras
# from bio import AlignIO
# from Bio import AlignIO
# from Bio.Align import MultipleSeqAlignment
# from Bio.SeqRecord import SeqRecord
# from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing import sequence
from sklearn.metrics import roc_auc_score
import sklearn
import numpy as np
import re
import pickle
import itertools
import random
import string
from tqdm import tqdm
import pandas as pd
# import dask.dataframe as pd
from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader, MixedLoader
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from spektral.layers import GCNConv, GlobalSumPool, AGNNConv, GATConv

from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.gcn_filter import GCNFilter
# model.predict(dataset[1])
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

print(tf.__version__)

le = LabelEncoder()
le.fit(['A', 'C', 'G', 'T', 'N', '-'])
print(le.transform(['A', 'C', 'G', 'T', 'N', '-']))
print(list(le.classes_))
sns.set()

#Initialize the graph
G = nx.Graph(name='G')

transcriptionFactor = sys.argv[1]
celltype = sys.argv[2]
gpu = int(sys.argv[3])

#Create nodes

#Each node is assigned node feature which corresponds to the node name
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
for a,b in enumerate(names):
    G.add_node(b, name=a)
# for i in range(5):
#     G.add_node(i, name=i)
#edges
edges = [('hg38','_HP'),('panTro4','_HP'),('gorGor3','_HPG'),('ponAbe2','_HPGP'),('_HP','_HPG'), ('_HPG','_HPGP'), ('nomLeu3', '_HPGPN'), 
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

# print('Graph Info:\n', nx.info(G))

# print('\nGraph Nodes: ', G.nodes.data())

#Plot the graph
# plt.figure(figsize=(15,15)) 
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
# print(len(names), len(edges))

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
            x = np.expand_dims(le.transform(self.rawdf.iloc[idx-self.margin][2:-1]), axis=1)
#             print(x)
            for ind in range(idx -self.margin+1, idx + self.margin +1):
                x = np.append(x, np.expand_dims(le.transform(self.rawdf.iloc[ind][2:-1]), axis=1), axis = 1) 
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
        return [make_graph(i) for i in tqdm(range(self.n_samples))]


import os
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, Input, Reshape, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())
import tensorflow as tf
# from spektral.models.general_gnn import GeneralGNN
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.layers import Attention, Dense, Input, Dropout, LSTM, Flatten,  Embedding, Attention, Reshape, Bidirectional, Conv1D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.saved_model import save, load
import os
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu)


from Models.cnn_bilstm_attention import RNATracker
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

hidden = 64
# from sklearn.model_selection import train_test_split
# chromosomes_train = [2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,22]
# chromosomes_val = []
# X_train = np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,2,transcriptionFactor,   celltype))
# y_train = np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,2,transcriptionFactor,   celltype))
# # X_val = np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(tf,2,tf,   celltype))
# # y_val = np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(tf,2,tf,   celltype))
# for chrom in chromosomes_train:
#     print(chrom)
#     X_train = np.concatenate((X_train, np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype))), axis=0)
#     y_train = np.concatenate((y_train, np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype))), axis=0)
# # for chrom in chromosomes_val:
# #     X_val = np.concatenate((X_train, np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(tf,chrom,tf,   celltype))), axis=0)
# #     y_val = np.concatenate((X_train, np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(tf,chrom,tf,   celltype))), axis=0)
from sklearn.model_selection import train_test_split
X_train = np.load('graphs/{}/X_revCompConcatenatedTrue_{}_badAlnRemoved.npy'.format(transcriptionFactor,celltype))
y_train = np.load('graphs/{}/y_revCompConcatenated_{}.npy'.format(transcriptionFactor,celltype))
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# X_test = np.load('graphs/dataset_101_chrAll_ccre_nonccre_{}_X_test.npy'.format(celltype))
# y_test = np.load('graphs/dataset_101_chrAll_ccre_nonccre_{}_y_test.npy'.format(celltype))
# print(X_train.shape)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False, stratify = None)
X_train = X_train[:,0,:]
X_test_rna = X_test
max_len = 402
nb_classes =2 
outpath = 'kerasModels/{}_{}_siamese_'.format(transcriptionFactor, celltype)
outpath_FactorNet = 'kerasModels/{}_{}_FactorNet_siamese_'.format(transcriptionFactor, celltype)
kfold_index = 2
nb_filters = 32
filters_length = 10
pooling_size = 3
lstm_unit = 32
nb_epochs =30
batch_size = 256
model = RNATracker(max_len, nb_classes, outpath, kfold_index)
model.build_model_advanced_masking(nb_filters, filters_length, pooling_size, lstm_unit, np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]))
model.train(X_train, y_train, batch_size, nb_epochs)

FactorNet = RNATracker(max_len, nb_classes, outpath_FactorNet, kfold_index)
FactorNet.build_model_advanced_masking(nb_filters, filters_length, pooling_size, lstm_unit, np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]))
FactorNet.train(X_train, y_train, batch_size, 70)

def draw_prc(model, X_test, y_test, name, celltype, baseline = False) :
    if baseline :
        y_score = model.predict(X_test[:,0,:])
    else :
        y_score = model.predict(X_test, batch_size = 256)
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = 2
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")
    print('AUPRC {} : {}'
          .format(name, average_precision["micro"]))
    plt.step(recall['micro'], precision['micro'], label = name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title(
#         'PRC curve CCRE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('figures/prc_ccre_{}_{}'.format(transcriptionFactor, celltype))
    return average_precision["micro"]

def draw_roc(model, X_test, y_test, name, celltype, baseline = False):
    if baseline :
        fpr, tpr, _ = roc_curve(y_test[:,1], model.predict(X_test[:,0,:])[:,1])
    else :
        fpr, tpr, _ = roc_curve(y_test[:,1], model.predict(X_test, batch_size = 256)[:,1])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    print('AUC {} :'.format(name), area)
    plt.savefig('figures/roc_ccre_{}_{}'.format(transcriptionFactor, celltype))
    return area

# model15_conv1d_H1 = tf.keras.models.load_model('kerasModels/model{}_conv1d_{}_{}_hidden{}_revCompConcatenated_siamese'.format(201, transcriptionFactor, celltype, hidden))
# model15_conv1d_H1 = tf.keras.models.load_model('kerasModels/model{}_conv3d_siamese_{}_{}_hidden{}_revCompConcatenated'.format(201, transcriptionFactor, celltype, hidden))
# model15_conv1d_H1 = tf.keras.models.load_model('kerasModels/model{}_conv3d_bahdanau_{}_{}_hidden{}'.format(201, transcriptionFactor, celltype, hidden))
# model15_conv1d_H1 = tf.keras.models.load_model('kerasModels/model{}_conv1d_siamese_{}_{}_hidden{}_revCompConcatenated_se'.format(201, transcriptionFactor, celltype, hidden))
# model15_conv1d_H1 = tf.keras.models.load_model('kerasModels/model{}_conv1d_siamese_{}_{}_hidden{}_revCompConcatenated_se_sigmoid'.format(201, transcriptionFactor, celltype, hidden))
model15_conv1d_H1 = tf.keras.models.load_model('kerasModels/model{}_conv1d_siamese_{}_{}_hidden{}_revCompConcatenated_se_badAlnRemoved'.format(201, transcriptionFactor, celltype, hidden))

# model15_conv1d_H1_linear = tf.keras.models.load_model('kerasModels/model{}_conv1d_siamese_{}_{}_hidden{}_revCompConcatenated_se_linear'.format(201, transcriptionFactor, celltype, hidden))



auroc_rna = draw_roc(model.model, X_test_rna, y_test, 'RNAtracker',celltype, baseline = True)
auroc_factor = draw_roc(FactorNet.model, X_test_rna, y_test, 'FactorNet',celltype, baseline = True)
# auroc_graphylo = draw_roc(model15_conv1d_H1, X_test, y_test, 'conv3d_{}_{}_sigmoid'.format(transcriptionFactor, celltype), celltype)
auroc_graphylo = draw_roc(model15_conv1d_H1, X_test, y_test, 'conv3d_{}_{}_sigmoid'.format(transcriptionFactor, celltype), celltype)
# auroc_graphylo = draw_roc(model15_conv1d_H1_linear, X_test, y_test, 'conv3d_{}_{}_linear'.format(transcriptionFactor, celltype), celltype)

auprc_rna = draw_prc(model.model, X_test_rna, y_test, 'RNAtracker', celltype, baseline = True)
auprc_factor =draw_prc(FactorNet.model, X_test_rna, y_test, 'FactorNet', celltype, baseline = True)
# auprc_graphylo =draw_prc(model15_conv1d_H1, X_test, y_test, 'conv3d_{}_{}_sigmoid'.format(transcriptionFactor, celltype), celltype)
auprc_graphylo =draw_prc(model15_conv1d_H1, X_test, y_test, 'conv3d_{}_{}_sigmoid'.format(transcriptionFactor, celltype), celltype)
# auroc_graphylo = draw_roc(model15_conv1d_H1_linear, X_test, y_test, 'conv3d_{}_{}_linear'.format(transcriptionFactor, celltype), celltype)

# print()
# print('{} {} Improvement in AUROC from RNAtracker : '.format(transcriptionFactor, celltype), auroc_graphylo - auroc_rna)
# print('{} {} Improvement in AUROC from FactorNet : '.format(transcriptionFactor, celltype), auroc_graphylo - auroc_factor)

# print()
# print('{} {} Improvement in AUPRC from RNAtracker : '.format(transcriptionFactor, celltype), auprc_graphylo - auprc_rna)
# print('{} {} Improvement in AUPRC from FactorNet : '.format(transcriptionFactor, celltype), auprc_graphylo - auprc_factor)
