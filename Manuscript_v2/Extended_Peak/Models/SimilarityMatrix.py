import scipy.io as sio
import numpy as np
from numpy.random import permutation
from sklearn import preprocessing

import sklearn.metrics.pairwise as pd

###############################################################################
# Define
###############################################################################
def LoadStackPeak(fileloc, name):
    '''Loads Matlab file and stacks data in proper format'''
    
    file = sio.loadmat(str(fileloc))
    PeakData = file[str(name)]
    return PeakData

def Standardize(dataset):
    """ Dataset format (Samples, Row, Col, Depth) """
    z_score = np.zeros(np.shape(dataset))
    for i in range(np.shape(dataset)[3]):
        z_score[:,:,:,i] = (dataset[:,:,:,i] - dataset[:,:,:,i].mean()) / dataset[:,:,:,i].std()
    
    return z_score

###############################################################################
# Load Data
###############################################################################
WAT = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/WAT_dataset.mat', 'WAT'))
BAT = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/BAT_dataset.mat', 'BAT'))
MUS = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/MUS_dataset.mat', 'MUS'))

WAT = WAT[16, 16, :, :]
BAT = BAT[16, 16, :, :]
MUS = MUS[16, 16, :, :]


# Make sizes equal
min_sample_size = np.min([np.shape(WAT)[0], np.shape(BAT)[0], np.shape(MUS)[0]])

# Randomize data
WAT = WAT[permutation(np.shape(WAT)[0]), :]
BAT = BAT[permutation(np.shape(BAT)[0]), :]
MUS = MUS[permutation(np.shape(MUS)[0]), :]

# Make classes equal
WAT = WAT[0:min_sample_size, :]
BAT = BAT[0:min_sample_size, :]
MUS = MUS[0:min_sample_size, :]

#### Run Cosine Similarity

#WATvWAT = np.sum(pd.cosine_similarity(WAT, Y=WAT)) / min_sample_size**2
#BATvBAT = np.sum(pd.cosine_similarity(BAT, Y=BAT)) / min_sample_size**2
#MUSvMUS = np.sum(pd.cosine_similarity(MUS, Y=MUS)) / min_sample_size**2
#
#WATvBAT = np.sum(pd.cosine_similarity(WAT, Y=BAT)) / min_sample_size**2
#WATvMUS = np.sum(pd.cosine_similarity(WAT, Y=MUS)) / min_sample_size**2
#
#BATvMUS = np.sum(pd.cosine_similarity(BAT, Y=MUS)) / min_sample_size**2
#
#WATvWAT2 = (np.sum(pd.cosine_similarity(WAT, Y=WAT))*.05) / min_sample_size

WATvWAT = pd.cosine_similarity(WAT, Y=WAT)