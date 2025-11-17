from sklearn.preprocessing import LabelEncoder
# from Bio import AlignIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn
import numpy as np
import re
import pickle
import itertools
import random
import string
from tqdm import tqdm
import pandas as pd
from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader, MixedLoader
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, MaxPooling2D, MaxPooling1D, MaxPooling3D, GlobalAveragePooling1D, GlobalAveragePooling2D, MultiHeadAttention
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from spektral.layers import GCNConv, GlobalSumPool, AGNNConv, GATConv
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.gcn_filter import GCNFilter
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, Input, Reshape, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
# from alibi.explainers import IntegratedGradients
# import matplotlib.pyplot as plt
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.layers import Attention, Dense, Input, Dropout, LSTM, Flatten,  Embedding, Attention, Reshape, Bidirectional, Conv1D, Conv2D, AdditiveAttention, multiply, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.saved_model import save, load
import os
import sys

data_path = sys.argv[1]
model_path = sys.argv[2]
target_path = sys.argv[3]
gpu = int(sys.argv[4])
num_filter = int(sys.argv[5])
num_hidden = int(sys.argv[6])
num_hidden_graph = int(sys.argv[7])


le = LabelEncoder()
le.fit(['A', 'C', 'G', 'T', 'N', '-'])
print(le.transform(['A', 'C', 'G', 'T', 'N', '-']))
print(list(le.classes_))
# sns.set()

# #Initialize the graph
G = nx.Graph(name='G')
os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu}"
# #Create nodes

# #Each node is assigned node feature which corresponds to the node name
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


#Plot the graph
# plt.figure(figsize=(15,15)) 
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
print(len(names), len(edges))

A = np.array(nx.attr_matrix(G, node_attr='name')[0])
print(A)
# class ModDataset(Dataset):

#     def __init__(self,seqdf, rawdf, a, margin, **kwargs):
#         self.rawdf = rawdf
#         self.seqdf = seqdf
#         self.margin = margin
#         self.a = a
#         self.indices = seqdf.index
#         self.n_samples = self.seqdf.shape[0]
#         super().__init__(**kwargs)

#     def read(self):
#         def make_graph(i):
#             n = self.seqdf.shape[0]

#             # Node features
#             idx = self.indices[i] #+ self.margin
# #             print(rawdf.loc[idx])
#             x = np.expand_dims(le.transform(self.rawdf.iloc[idx-self.margin][2:-1]), axis=1)
# #             print(x)
#             for ind in range(idx -self.margin+1, idx + self.margin +1):
#                 x = np.append(x, np.expand_dims(le.transform(self.rawdf.iloc[ind][2:-1]), axis=1), axis = 1) 
# #                 x = x+ le.transform(self.rawdf.iloc[ind][1:-1])
#             x = list(x)
# #             print(idx, x)
#             x = np.asarray(x).astype('float32')
# #             print(x.shape)
            
#             y = self.rawdf['y'].loc[idx]
#             mat = np.zeros((2))
#             mat[y] = 1
#             y = mat

#             # Edges
#             I = np.identity(A.shape[0])
#             AI = A #+I
#             a = sp.csr_matrix(AI)

#             return Graph(x=x, a=a, y=y)

#         # We must return a list of Graph objects
#         return [make_graph(i) for i in tqdm(range(self.n_samples))]




from sklearn.model_selection import train_test_split
# X_train = np.load('graphs/{}/X_revCompIncluded_{}.npy'.format(transcriptionFactor,celltype))
# y_train = np.load('graphs/{}/y_revCompIncluded_{}.npy'.format(transcriptionFactor,celltype))
# X_train = np.load('graphs/{}/X_revCompConcatenated_{}.npy'.format(transcriptionFactor,celltype))
# y_train = np.load('graphs/{}/y_revCompConcatenated_{}.npy'.format(transcriptionFactor,celltype))
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
# chromosomes_train = [2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,22]
# chromosomes_val = []
# X_train = np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,2,transcriptionFactor,   celltype))
# y_train = np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,2,transcriptionFactor,   celltype))
# for chrom in chromosomes_train:
#     print(chrom)
#     X_train = np.concatenate((X_train, np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype))), axis=0)
#     y_train = np.concatenate((y_train, np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype))), axis=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False, stratify = None)
def species_attention(input, ratio=3):
    residual = input
    try :
        se = GlobalAveragePooling2D(data_format='channels_first')(residual)
    except :
        se = GlobalAveragePooling1D(data_format='channels_first')(residual)
    se = Reshape((1,1,residual.shape[1]))(se)
    print(se.shape)
    se = Dense(residual.shape[1] // ratio, activation='relu', use_bias=False)(se)
    se = Dropout(.3)(se)
    se = Dense(residual.shape[1] // ratio, activation='relu', use_bias=False)(se)
    se = Dropout(.3)(se)
    se = Dense(residual.shape[1], activation='sigmoid', use_bias=False)(se)
    se = Permute((3, 1, 2))(se)
    print(residual.shape, se.shape)
    x = multiply([residual, se])
    return x

def channel_attention(input, ratio=3):
    residual = input
    try :
        se = GlobalAveragePooling2D()(residual)
    except :
        se = GlobalAveragePooling1D()(residual)
#     se = Reshape((1,1,residual.shape[1]))(se)
    print(se.shape)
    se = Dense(residual.shape[-1] // ratio, activation='relu', use_bias=False)(se)
    se = Dropout(.3)(se)
    se = Dense(residual.shape[-1] // ratio, activation='relu', use_bias=False)(se)
    se = Dropout(.3)(se)
    se = Dense(residual.shape[-1], activation='sigmoid', use_bias=False)(se)
    print(residual.shape, se.shape)
    x = multiply([residual, se])
    return x

def spatial_attention(input_feature, kernel_size=7):
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(input_feature)
    spatial = avg_pool
    print('This')
    print(avg_pool.shape)
    spatial = Conv2D(1, (kernel_size, kernel_size), strides=1, activation='sigmoid', padding='same', data_format='channels_first')(spatial)
    result = multiply([input_feature, spatial])
    return result


def model_siamese_each(inputs_l):    
    x = inputs_l
    x = Conv2D(32, (10,6), padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling3D((2, 2, 2), data_format='channels_first')(x)
    return x

def model_siamese_onehot(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]))
    inputs_l, inputs_r = tf.split(inputs, [length,length], 2)
    inputs_l = tf.expand_dims(inputs_l, -1)
    inputs_r = tf.expand_dims(inputs_r, -1)
    
    x = model_siamese_each(inputs_l)
    x_right = model_siamese_each(inputs_r)
    x_right = tf.reverse(x_right, [2])
    x = tf.add(x, x_right)
    print(x.shape)
    x = Conv2D(32, (10,3), padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling3D((2, 2, 2), data_format='channels_first')(x)
    print(x.shape)

    
    x = Reshape(( x.shape[1], -1))(x)
    x = GCNConv(32, activation="relu")([x,A])
    x = Dropout(.3)(x)
#     x = GCNConv(32, activation="relu")([x,A])
#     x = Dropout(.3)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def model_siamese(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]), dtype='uint8')
    x = tf.one_hot(inputs, 6)
    inputs_l, inputs_r = tf.split(x, [length,length], 2)
    inputs_l = tf.expand_dims(inputs_l, -1)
    inputs_r = tf.expand_dims(inputs_r, -1)
    
    x = model_siamese_each(inputs_l)
    x_right = model_siamese_each(inputs_r)
    x_right = tf.reverse(x_right, [2])
    x = tf.add(x, x_right)
    print(x.shape)
    x = Conv2D(32, (10,3), padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling3D((2, 2, 2), data_format='channels_first')(x)
    print(x.shape)

    
    x = Reshape(( x.shape[1], -1))(x)
    x = GCNConv(32, activation="relu")([x,A])
    x = Dropout(.3)(x)
#     x = GCNConv(32, activation="relu")([x,A])
#     x = Dropout(.3)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def model_siamese1D_each(inputs_l):    
    x = inputs_l
    x = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    return x

def model_siamese1D_onehot(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]))
    inputs_l, inputs_r = tf.split(inputs, [length,length], 2)
    
    shared_conv = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=inputs_l.shape[2:])
    x = shared_conv(inputs_l)
    x_right = shared_conv(inputs_r)
    x_right = tf.reverse(x_right, [2])
    # x = model_siamese1D_each(inputs_l)
    # x_right = model_siamese1D_each(inputs_r)
    # x_right = tf.reverse(x_right, [2])
#     x = tf.add(x, x_right)
    x = tf.concat([x, x_right], -1)
    print(x.shape)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    print(x.shape)
    x = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    x = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    print(x.shape)

    
    x = Reshape(( x.shape[1], -1))(x)
    x = GCNConv(32, activation="relu")([x,A])
    x = Dropout(.3)(x)
    x = GCNConv(32, activation="relu")([x,A])
    x = Dropout(.3)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def model_siamese1D(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]), dtype='uint8')
    x = tf.one_hot(inputs, 6)
    inputs_l, inputs_r = tf.split(x, [length,length], 2)

    shared_conv = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=inputs_l.shape[2:])
    x = shared_conv(inputs_l)
    x_right = shared_conv(inputs_r)
    x_right = tf.reverse(x_right, [2])    
    # x = model_siamese1D_each(inputs_l)
    # x_right = model_siamese1D_each(inputs_r)
    # x_right = tf.reverse(x_right, [2])
#     x = tf.add(x, x_right)
    x = tf.concat([x, x_right], -1)
    print(x.shape)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    print(x.shape)
    x = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    x = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    print(x.shape)

    
    x = Reshape(( x.shape[1], -1))(x)
    x = GCNConv(32, activation="relu")([x,A])
    x = Dropout(.3)(x)
    x = GCNConv(32, activation="relu")([x,A])
    x = Dropout(.3)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def model_conv3d_siamese(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]), dtype='uint8')
    x = tf.one_hot(inputs, 6)
    inputs_l, inputs_r = tf.split(x, [length,length], 2)
    
    shared_conv = Conv2D(115, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    x = shared_conv(inputs_l)
    x_right = shared_conv(inputs_r)
    x_right = tf.reverse(x_right, [2])
    x = tf.concat([x, x_right], -1)
    print(x.shape)
    x = MaxPooling2D((2, 2), data_format='channels_first')(x)
    print(x.shape)
    x = Conv2D(115, (10,6), padding = 'same', activation='relu', data_format='channels_first')(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2, 2), data_format='channels_first')(x)
    print(x.shape)
    
    x = Reshape(( x.shape[1], -1))(x)
    print(x.shape)
    x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])
    x = Dropout(.3)(x)
    x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])
    x = Dropout(.3)(x)

    x = Flatten()(x)
    x = Dense(hidden, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def model_conv3d_bahdanau(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]), dtype='uint8')
    x = tf.one_hot(inputs, 6)
    inputs_l, inputs_r = tf.split(x, [length,length], 2)
    
    shared_k = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    shared_q = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    shared_v = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
#     x_k = shared_k(inputs_l)
#     x_right_k = shared_k(inputs_r)
#     x_right_k = tf.reverse(x_right_k, [2])
#     x_k = tf.concat([x_k, x_right_k], -1)
    
    x_q = shared_q(inputs_l)
    x_q = Dropout(.3)(x_q)
    x_right_q = shared_q(inputs_r)
    x_right_q = Dropout(.3)(x_right_q)
    x_right_q = tf.reverse(x_right_q, [2])
    x_q = tf.concat([x_q, x_right_q], -1)
    x_q = MaxPooling2D((2, 2), data_format='channels_first')(x_q)
    
    x_v = shared_v(inputs_l)
    x_v = Dropout(.3)(x_v)
    x_right_v = shared_v(inputs_r)
    x_right_v = Dropout(.3)(x_right_v)
    x_right_v = tf.reverse(x_right_v, [2])
    x_v = tf.concat([x_v, x_right_v], -1)
    x_v = MaxPooling2D((2, 2), data_format='channels_first')(x_v)
    x = AdditiveAttention()([x_q, x_v])
    
    x = tf.keras.layers.Concatenate()(
    [x_q, x])
    
    x = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first')(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2, 2), data_format='channels_first')(x)
    
#     x = Reshape(( x.shape[1], -1))(x)
    
#     print(x.shape)
#     x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])
#     x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model_attention = Model(inputs=inputs, outputs=outputs)
    model_attention.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model_attention

def model_conv3d_bahdanau_onehot(X_train_shape):
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]))
    inputs_l, inputs_r = tf.split(inputs, [length,length], 2)
    
    shared_k = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    shared_q = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    shared_v = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
#     x_k = shared_k(inputs_l)
#     x_right_k = shared_k(inputs_r)
#     x_right_k = tf.reverse(x_right_k, [2])
#     x_k = tf.concat([x_k, x_right_k], -1)
    
    x_q = shared_q(inputs_l)
    x_q = Dropout(.3)(x_q)
    x_right_q = shared_q(inputs_r)
    x_right_q = Dropout(.3)(x_right_q)
    x_right_q = tf.reverse(x_right_q, [2])
    x_q = tf.concat([x_q, x_right_q], -1)
    x_q = MaxPooling2D((2, 2), data_format='channels_first')(x_q)
    
    x_v = shared_v(inputs_l)
    x_v = Dropout(.3)(x_v)
    x_right_v = shared_v(inputs_r)
    x_right_v = Dropout(.3)(x_right_v)
    x_right_v = tf.reverse(x_right_v, [2])
    x_v = tf.concat([x_v, x_right_v], -1)
    x_v = MaxPooling2D((2, 2), data_format='channels_first')(x_v)
    x = AdditiveAttention()([x_q, x_v])
    
    x = tf.keras.layers.Concatenate()(
    [x_q, x])
    
    x = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first')(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2, 2), data_format='channels_first')(x)
    
#     x = Reshape(( x.shape[1], -1))(x)
    
#     print(x.shape)
#     x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])
#     x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model_attention = Model(inputs=inputs, outputs=outputs)
    model_attention.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model_attention

def model_conv3d_bahdanau_onehot_human(X_train_shape):
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]))
    inputs_l, inputs_r = tf.split(inputs, [length,length], 2)
    
    shared_k = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    shared_q = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
    shared_v = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first', input_shape=inputs_l.shape[1:])
#     x_k = shared_k(inputs_l)
#     x_right_k = shared_k(inputs_r)
#     x_right_k = tf.reverse(x_right_k, [2])
#     x_k = tf.concat([x_k, x_right_k], -1)
    
    x_q = shared_q(inputs_l)
    x_q = Dropout(.3)(x_q)
    x_right_q = shared_q(inputs_r)
    x_right_q = Dropout(.3)(x_right_q)
    x_right_q = tf.reverse(x_right_q, [2])
    x_q = tf.concat([x_q, x_right_q], -1)
    x_q = MaxPooling2D((2, 2), data_format='channels_first')(x_q)
    
    x_v = shared_v(inputs_l)
    x_v = Dropout(.3)(x_v)
    x_right_v = shared_v(inputs_r)
    x_right_v = Dropout(.3)(x_right_v)
    x_right_v = tf.reverse(x_right_v, [2])
    x_v = tf.concat([x_v, x_right_v], -1)
    x_v = MaxPooling2D((2, 2), data_format='channels_first')(x_v)
    x = AdditiveAttention()([x_q, x_v])
    
    x = tf.keras.layers.Concatenate()(
    [x_q, x])
    
    x = Conv2D(32, (10,6), padding = 'same', activation='relu', data_format='channels_first')(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2, 2), data_format='channels_first')(x)
    
#     x = Reshape(( x.shape[1], -1))(x)
    
#     print(x.shape)
#     x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])
#     x = GCNConv(32, activation="relu", kernel_regularizer=l2(5e-4))([x,A])

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model_attention = Model(inputs=inputs, outputs=outputs)
    model_attention.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model_attention


def model1D_siamese_onehot_se(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    input = Input(shape=(X_train_shape[1:]))
    x = species_attention(input)
    x = spatial_attention(x)
    inputs_l, inputs_r = tf.split(x, [length,length], 2)
    print(inputs_r.shape)
    shared_conv = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=inputs_l.shape[2:])
    shared_conv2 = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=inputs_l.shape[2:])
    x = shared_conv(inputs_l)
    x = species_attention(x)
    x = channel_attention(x)
    print(x.shape)
    x_right = shared_conv2(inputs_r)
    x = species_attention(x)
    x = channel_attention(x)
    x_right = tf.reverse(x_right, [2])
    x = tf.keras.layers.Concatenate(axis = -1)([x, x_right])
    print(x.shape)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    print(x.shape)
    x = Conv1D(32, 10, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = species_attention(x)
    x = channel_attention(x)
    x = MaxPooling2D((2, 1), data_format='channels_first')(x)
    print(x.shape)

    
#     x = Reshape(( x.shape[1], -1))(x)
#     x = GCNConv(32, activation="relu")([x,A])
#     x = Dropout(.3)(x)
#     x = GCNConv(32, activation="relu")([x,A])
#     x = Dropout(.3)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(2, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=input, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def model1D_siamese_se(X_train_shape):
    tf.keras.backend.clear_session()
    length = int(X_train_shape[2]/2)
    inputs = Input(shape=(X_train_shape[1:]), dtype='uint8')
    x = tf.one_hot(inputs, 6)
    inputs_l, inputs_r = tf.split(x, [length,length], 2)
    print(inputs_r.shape)
    shared_conv = Conv1D(num_filter, 11, padding = 'same', activation='relu', input_shape=inputs_l.shape[2:])
    shared_conv2 = Conv1D(num_filter, 11, padding = 'same', activation='relu', input_shape=inputs_l.shape[2:])
    x = shared_conv(inputs_l)
    x = channel_attention(x)
    print(x.shape)
    x_right = shared_conv2(inputs_r)
    x = channel_attention(x)
    x_right = tf.reverse(x_right, [2])
    x = tf.concat([x, x_right], -1)
    print(x.shape)
    x = MaxPooling2D((2,1), data_format='channels_first')(x)
    print(x.shape)
    x = Conv1D(num_filter, 11, padding = 'same', activation='relu', input_shape=x.shape[2:])(x)
    x = species_attention(x)
    x = Dropout(.3)(x)
    x = MaxPooling2D((2, 1), data_format='channels_first')(x)
    print(x.shape)

    
    x = Reshape((x.shape[1], -1))(x)
    x = GCNConv(num_hidden_graph, activation="relu")([x,A])
    x = Dropout(.3)(x)
    x = GCNConv(num_hidden_graph, activation="relu", kernel_regularizer=l2(5e-4))([x,A])
    x = Dropout(.3)(x)
    print(x.shape)
    x = Flatten()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(1, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',#BinaryFocalLoss(gamma=2),#'categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


from sklearn.model_selection import train_test_split

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes, dtype=np.uint8)[np.array(targets).reshape(-1)].astype('uint8')
    return res.reshape(list(targets.shape)+[nb_classes])
X_train = np.load(data_path).astype('uint8')
y_train = np.load(target_path)
# X_train_onehot = get_one_hot(X_train, 6)
# np.save('graphs/{}/X_revCompConcatenated_onehot_{}.npy'.format(transcriptionFactor,celltype), X_train_onehot)

print(X_train.shape, y_train.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# X_val, X_test,  y_val, y_test = train_test_split(X_val,  y_val, test_size=0.5, random_state=42)


def modelFit(filepath, epoch, batchSize, X_train, y_train, X_val, y_val, model):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    
    model.summary()
    hist1 = model.fit(X_train,
          y_train,
          epochs=epoch,
          callbacks=[callback, checkpoint],
          batch_size=batchSize,
          verbose=1,
          validation_data = (X_val, y_val)
          )
    plt.plot(hist1.history['loss'])
    plt.plot(hist1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{model_path}_learningCurve.png')
    plt.show()
    return hist1, model

hidden_sizes = [64]
for hidden in hidden_sizes:
    print(hidden)
    model1d_siamese_se= model1D_siamese_se(X_train.shape)
    # model1d_siamese_onehot_se= model1D_siamese_onehot_se(X_train_onehot.shape)
    # hist, model1d_siamese_onehot_se = modelFit('kerasModels/model{}_conv1d_siamese_{}_{}_hidden{}_revCompConcatenated_onehot_se'.format(201,transcriptionFactor, celltype, 64), 9000, 64, X_train_onehot, y_train, X_val_onehot, y_val, model1d_siamese_onehot_se)
    hist, model1d_siamese_se = modelFit('{}'.format(model_path), 9000, 64, X_train, y_train, X_val, y_val, model1d_siamese_se)

    # model_conv3d = model_conv3d_siamese(X_train)
    # hist_conv3d,model15_conv3d = modelFit('kerasModels/model{}_conv3d_siamese_{}_{}_hidden{}_revCompConcatenated'.format(201,transcriptionFactor, celltype, hidden), 9000, 256, X_train, y_train, X_val, y_val, model_conv3d)
    
    # model_conv3d_bahdanau = model_conv3d_bahdanau(X_train.shape)
    # model_conv3d_bahdanau_onehot = model_conv3d_bahdanau_onehot(X_train_onehot.shape)
    # model_conv3d_bahdanau_onehot_human = model_conv3d_bahdanau_onehot_human(X_train_onehot_human.shape)
    # hist_conv3d_bahdanau_onehot, model_conv3d_bahdanau_onehot_human = modelFit('kerasModels/model{}_conv3d_bahdanau_{}_{}_hidden{}_onehot_human'.format(201,transcriptionFactor, celltype, hidden), 9000, 32, X_train_onehot_human, y_train, X_val_onehot_human, y_val, model_conv3d_bahdanau_onehot_human)
    # hist_conv3d_bahdanau, model_conv3d_bahdanau = modelFit('kerasModels/model{}_conv3d_bahdanau_{}_{}_hidden{}'.format(201,transcriptionFactor, celltype, hidden), 9000, 32, X_train, y_train, X_val, y_val, model_conv3d_bahdanau)
    # hist_conv3d_bahdanau_onehot, model_conv3d_bahdanau_onehot = modelFit('kerasModels/model{}_conv3d_bahdanau_{}_{}_hidden{}_onehot'.format(201,transcriptionFactor, celltype, hidden), 9000, 32, X_train_onehot, y_train, X_val_onehot, y_val, model_conv3d_bahdanau_onehot)

# model_siamese = model_siamese1D(X_train)
# hist, model_siamese = modelFit('kerasModels/model{}_conv1d_{}_{}_hidden{}_revCompConcatenated_siamese'.format(201,transcriptionFactor, celltype, 64), 9000, 64, X_train, y_train, X_val, y_val, model_siamese)

# X_train = [1]

# X_train_onehot = np.load('graphs/{}/X_revCompConcatenated_onehot_{}.npy'.format(transcriptionFactor,celltype))
# y_train = np.load('graphs/{}/y_revCompConcatenated_{}.npy'.format(transcriptionFactor,celltype))
# print(X_train_onehot.shape)

# X_train_onehot, X_val_onehot, y_train, y_val = train_test_split(X_train_onehot, y_train, test_size=0.4, random_state=42)
# X_val_onehot, X_test_onehot, y_val, y_test = train_test_split(X_val_onehot , y_val, test_size=0.5, random_state=42)

# print(X_train_onehot.shape)
# print(X_train_onehot[:,:,int(X_train_onehot.shape[2]/2):,:].shape)
# model_siamese_onehot = model_siamese1D_onehot(X_train_onehot)
# hist, model_siamese_onehot = modelFit('kerasModels/model{}_conv1d_{}_{}_hidden{}_revCompConcatenated_siamese_onehot'.format(201,transcriptionFactor, celltype, 64), 9000, 64, X_train_onehot, y_train,X_val_onehot, y_val, model_siamese_onehot)
# #0.289 for same 2 layer conv2d 2 layer regularized gcn
