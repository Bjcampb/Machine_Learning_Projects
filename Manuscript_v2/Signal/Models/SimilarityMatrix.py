import numpy as np
from numpy.random import permutation
from sklearn import preprocessing

import sklearn.metrics.pairwise as pd

###############################################################################
# Load Data
###############################################################################
WAT = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_WAT.csv', delimiter=',')
BAT = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_BAT.csv', delimiter=',')
MUS = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_MUS.csv', delimiter=',')

# Make sizes equal
min_sample_size = np.min([np.shape(WAT)[0], np.shape(BAT)[0], np.shape(MUS)[0]])

# Randomize data
WAT = WAT[permutation(np.shape(WAT)[0]), :]
BAT = BAT[permutation(np.shape(BAT)[0]), :]
MUS = MUS[permutation(np.shape(MUS)[0]), :]

# Make classes equal
WAT = WAT[0:min_sample_size, 0:24]
BAT = BAT[0:min_sample_size, 0:24]
MUS = MUS[0:min_sample_size, 0:24]

X_train = np.concatenate((WAT, BAT, MUS), axis = 0)


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

WAT = X_train[0:min_sample_size, :]
BAT = X_train[min_sample_size:min_sample_size*2, :]
MUS = X_train[min_sample_size*2:, :]

#### Run Cosine Similarity

WATvWAT = np.sum(pd.cosine_similarity(WAT, Y=WAT)) / min_sample_size**2
BATvBAT = np.sum(pd.cosine_similarity(BAT, Y=BAT)) / min_sample_size**2
MUSvMUS = np.sum(pd.cosine_similarity(MUS, Y=MUS)) / min_sample_size**2

WATvBAT = np.sum(pd.cosine_similarity(WAT, Y=BAT)) / min_sample_size**2
WATvMUS = np.sum(pd.cosine_similarity(WAT, Y=MUS)) / min_sample_size**2

BATvMUS = np.sum(pd.cosine_similarity(BAT, Y=MUS)) / min_sample_size**2
