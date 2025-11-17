import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

transcriptionFactor = sys.argv[1]
celltype = sys.argv[2]

transcriptionFactors = [transcriptionFactor]
celltypes = [celltype]
def reverseComplement(input):
    if int(input) == 1:
        return 5
    elif int(input) == 5:
        return 1
    elif int(input) == 2:
        return 3
    elif (input) == 3:
        return 2
    else:
        return input

def makeReverse(input):
    output = np.zeros(input.shape)
    a,b,c = input.shape
    for i in tqdm(range(a)):
        for j in range(b):
            for k in range(c):
                output[i,j,k] = reverseComplement(input[i,j,k])
    return np.flip(output, axis=2)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes, dtype=np.uint8)[np.array(targets).reshape(-1)].astype('uint8')
    return res.reshape(list(targets.shape)+[nb_classes])

# for transcriptionFactor in transcriptionFactors:
#     for celltype in celltypes: 
#         try :
#             encode_list = ['-', 'A', 'C', 'G', 'N', 'T']
#             chromosomes_train = [2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,22]
#             chromosomes_val = []
#             for chrom in tqdm(chromosomes_train):
#                 X_train = np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype)).astype(np.uint8)
#                 y_train = np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype)).astype(np.uint8)
#                 # X_train301 = np.load('graphs/{}/dataset_301_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype)).astype(np.uint8)
#                 # y_train301 = np.load('graphs/{}/dataset_301_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype)).astype(np.uint8)
#                 # X_train401 = np.load('graphs/{}/dataset_401_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype)).astype(np.uint8)
#                 # y_train401 = np.load('graphs/{}/dataset_401_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype)).astype(np.uint8)
#                 np.save('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype), X_train.astype(np.uint8))
#                 np.save('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype), y_train.astype(np.uint8))
#                 # np.save('graphs/{}/dataset_301_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype), X_train301.astype(np.uint8))
#                 # np.save('graphs/{}/dataset_301_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype), y_train301.astype(np.uint8))
#                 # np.save('graphs/{}/dataset_401_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype), X_train401.astype(np.uint8))
#                 # np.save('graphs/{}/dataset_401_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,chrom,transcriptionFactor,   celltype), y_train401.astype(np.uint8))
#         except:
#             print('error')
#             continue
lengths = [1001]
for transcriptionFactor in transcriptionFactors:
    for celltype in celltypes: 
        for length in lengths :
            encode_list = ['-', 'A', 'C', 'G', 'N', 'T']
            chromosomes_train = [3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,22]
            chromosomes_val = []
            X_train = np.load('graphs/{}/dataset_{}_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,length,2,transcriptionFactor,   celltype)).astype(np.uint8)
            y_train = np.load('graphs/{}/dataset_{}_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,length,2,transcriptionFactor,   celltype)).astype(np.uint8)
            # X_val = np.load('graphs/{}/dataset_201_chr{}_{}_{}_X_train.npy'.format(tf,2,tf,   celltype))
            # y_val = np.load('graphs/{}/dataset_201_chr{}_{}_{}_y_train.npy'.format(tf,2,tf,   celltype))
            for chrom in tqdm(chromosomes_train):
    #                 print(chrom)
                X_train = np.concatenate((X_train, np.load('graphs/{}/dataset_{}_chr{}_{}_{}_X_train.npy'.format(transcriptionFactor,length,chrom,transcriptionFactor,   celltype)).astype(np.uint8)), axis=0)
                y_train = np.concatenate((y_train, np.load('graphs/{}/dataset_{}_chr{}_{}_{}_y_train.npy'.format(transcriptionFactor,length,chrom,transcriptionFactor,   celltype)).astype(np.uint8)), axis=0)

            # result = makeReverse(X_train)
            myfunc_vec = np.vectorize(reverseComplement)
            result = myfunc_vec(X_train)
            # def makeReverse(input):
            print(transcriptionFactor,celltype)
            print(result.shape)
            print(type(result[0,0,0]))
            print(X_train.shape)
            print(type(X_train[0,0,0]))
            result =  np.flip(result, axis=2)
            X_train_revComp = np.concatenate((X_train, result), axis=2).astype(np.uint8)
            y_train = y_train.astype(np.uint8)
            print(X_train_revComp.shape, y_train.shape)  
            np.save('graphs/{}/X_revCompConcatenatedTrue{}_{}.npy'.format(transcriptionFactor,length,celltype), X_train_revComp)
            np.save('graphs/{}/y_revCompConcatenatedTrue{}_{}.npy'.format(transcriptionFactor,length,celltype), y_train)
            np.save('graphs/{}/y_revCompConcatenated{}_{}.npy'.format(transcriptionFactor,length,celltype), y_train)
            # X_train_onehot = get_one_hot(X_train_revComp, 6)
            # print(X_train_onehot.shape) 
            # np.save('graphs/{}/X_revCompConcatenated_onehot_{}.npy'.format(transcriptionFactor,celltype), X_train_onehot)
            # X_train_revComp = np.concatenate((X_train, result), axis=0).astype(np.uint8)
            # y_train_revComp = np.concatenate((y_train, y_train), axis=0).astype(np.uint8)
            # print(X_train_revComp.shape, y_train_revComp.shape)  
            # np.save('graphs/{}/X_revCompIncluded_{}.npy'.format(transcriptionFactor,celltype), X_train_revComp.astype(np.uint8))
            # np.save('graphs/{}/y_revCompIncluded_{}.npy'.format(transcriptionFactor,celltype), y_train_revComp.astype(np.uint8))

            X_train_revComp = []
            y_train_revComp = []
            X_train = []
            y_train = []
            result = []