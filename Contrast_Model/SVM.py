import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

########################################################
# Define
########################################################

def LoadMat(fileloc):
    '''
    Loads MatLab file and stacks data in proper format
    '''

    file = sio.loadmat(str(fileloc))
    PeakData = file['dataset']

    return PeakData

########################################################
# Load Data
########################################################

# August Surface Coil #

# Mouse 1
Surface_Mouse_1_IGWAT_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_1_pre_igwat_mus.mat')
Surface_Mouse_1_IGWAT_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_1_pre_igwat_wat.mat')
Surface_Mouse_1_INTER_BAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_1_pre_intercap_bat.mat')
Surface_Mouse_1_INTER_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_1_pre_intercap_mus.mat')
Surface_Mouse_1_INTER_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_1_pre_intercap_wat.mat')

# Mouse 2
Surface_Mouse_2_IGWAT_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_2_pre_igwat_mus.mat')
Surface_Mouse_2_IGWAT_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_2_pre_igwat_wat.mat')
Surface_Mouse_2_INTER_BAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_2_pre_intercap_bat.mat')
Surface_Mouse_2_INTER_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_2_pre_intercap_mus.mat')
Surface_Mouse_2_INTER_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'aug_mouse_2_pre_intercap_wat.mat')

# February Volume Coil #

# Mouse 1
Volume_Mouse_1_IGWAT_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_1_pre_igwat_mus.mat')
Volume_Mouse_1_IGWAT_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_1_pre_igwat_wat.mat')
Volume_Mouse_1_INTER_BAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_1_pre_intercap_bat.mat')
Volume_Mouse_1_INTER_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_1_pre_intercap_mus.mat')
Volume_Mouse_1_INTER_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_1_pre_intercap_wat.mat')

# Mouse 3
Volume_Mouse_3_IGWAT_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_3_pre_igwat_mus.mat')
Volume_Mouse_3_IGWAT_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_3_pre_igwat_wat.mat')

# Mouse 4
Volume_Mouse_4_INTER_BAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_4_pre_intercap_bat.mat')
Volume_Mouse_4_INTER_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_4_pre_intercap_mus.mat')
Volume_Mouse_4_INTER_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_4_pre_intercap_wat.mat')

# Mouse 5
Volume_Mouse_5_INTER_BAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_5_pre_intercap_bat.mat')
Volume_Mouse_5_INTER_MUS = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_5_pre_intercap_mus.mat')
Volume_Mouse_5_INTER_WAT = LoadMat('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/1D/'
                   'feb_mouse_5_pre_intercap_wat.mat')

###############################################################################
# Standardize and Organize
###############################################################################
min_max_scaler = preprocessing.MinMaxScaler()

WAT_training = np.transpose(np.hstack((Surface_Mouse_1_IGWAT_WAT, Surface_Mouse_1_INTER_WAT, 
                                       Volume_Mouse_1_IGWAT_WAT, Volume_Mouse_1_INTER_WAT, 
                                       Volume_Mouse_3_IGWAT_WAT, Volume_Mouse_4_INTER_WAT, 
                                       Volume_Mouse_5_INTER_WAT)))
WAT_test = np.transpose(np.hstack((Surface_Mouse_2_IGWAT_WAT, Surface_Mouse_2_INTER_WAT)))

BAT_training = np.transpose(np.hstack((Surface_Mouse_1_INTER_BAT, Volume_Mouse_1_INTER_BAT, 
                                       Volume_Mouse_4_INTER_BAT, Volume_Mouse_5_INTER_BAT)))
BAT_test = np.transpose(Surface_Mouse_2_INTER_BAT)

MUS_training = np.transpose(np.hstack((Surface_Mouse_1_IGWAT_MUS, Surface_Mouse_1_INTER_MUS, 
                                       Volume_Mouse_1_IGWAT_MUS, Volume_Mouse_1_INTER_MUS, 
                                       Volume_Mouse_3_IGWAT_MUS, Volume_Mouse_4_INTER_MUS, 
                                       Volume_Mouse_5_INTER_MUS)))
MUS_test = np.transpose(np.hstack((Surface_Mouse_2_IGWAT_MUS, Surface_Mouse_2_INTER_MUS)))

WAT_training = WAT_training[0:np.shape(MUS_training)[0],:]
BAT_training = BAT_training[0:np.shape(MUS_training)[0],:]

# Create classes
WAT_train_class = np.zeros((WAT_training.shape[0],1), dtype=int)
WAT_test_class = np.zeros((WAT_test.shape[0],1), dtype=int)

BAT_train_class = np.zeros((BAT_training.shape[0],1), dtype=int)+1
BAT_test_class = np.zeros((BAT_test.shape[0],1), dtype=int)+1

MUS_train_class = np.zeros((MUS_training.shape[0],1), dtype=int)+2
MUS_test_class = np.zeros((MUS_test.shape[0],1), dtype=int)+2

#############################################################################
## Create class
##############################################################################
X_train = np.concatenate((WAT_training, BAT_training, MUS_training), axis=0)
y_train = np.concatenate((WAT_train_class, BAT_train_class, MUS_train_class))
y_train = np.ravel(y_train)

X_test = np.concatenate((WAT_test, BAT_test, MUS_test), axis=0)
y_test = np.concatenate((WAT_test_class, BAT_test_class, MUS_test_class))
y_test = np.ravel(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train[:,7]
X_train = X_train.reshape(-1,1)
X_test = X_test[:,7]
X_test = X_test.reshape(-1,1)

# Normalize training and testing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


X_test = scaler.transform(X_test)


# Switch order of matrix dimensions to fix keras (expects (samples, steps, input_dim))
#(real/imag, samples, echos) - (samples, echos, real/imag)
#X_train = np.expand_dims(X_train, axis=0)
#X_test = np.expand_dims(X_test, axis=0)
#X_train = np.einsum('ijk -> jki', X_train)
#X_test = np.einsum('ijk -> jki', X_test)

# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]

#############################################################################
## Create SVM
###############################################################################
clf = SVC(C=35.0, gamma=.01, probability=True)
#clf = LinearSVC()
clf.fit(X_train, y_train)
Train_Accuracy = clf.fit(X_train, y_train).score(X_train, y_train)
test_prediction = clf.predict(X_test)
coefs = clf.dual_coef_

Test_accuracy = test_prediction == y_test
Test_accuracy = sum(Test_accuracy)
Test_accuracy = Test_accuracy / np.shape(y_test)[0]
Validation_Accuracy = clf.predict_proba(X_test)

##############################################################################
#                     Printing
###############################################################################
print('\nEcho Model SVM \n-----------------------')
print('Training Accuracy: %.2f \nValidation Accuracy:%.2f' % (Train_Accuracy*100, Test_accuracy*100))