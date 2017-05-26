import scipy.io as sio
import numpy as np
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.utils import np_utils
from keras import optimizers

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

# Normalize training and testing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


X_test = scaler.transform(X_test)


# Switch order of matrix dimensions to fix keras (expects (samples, steps, input_dim))
#(real/imag, samples, echos) - (samples, echos, real/imag)
X_train = np.expand_dims(X_train, axis=0)
X_test = np.expand_dims(X_test, axis=0)
X_train = np.einsum('ijk -> jki', X_train)
X_test = np.einsum('ijk -> jki', X_test)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

###############################################################################
##         Create the model
###############################################################################

# Signal Parameters
echos = 8    # Total amount of time steps
depth = 1  # Real and Imag have been combined
weight_decay = 0.0001#0.00001

num_filter = [32, 16]
length_filter = [4, 4]
pool_length = [2,2]

# CNN NETWORK
model = Sequential() # Initializes model to sequientially add layers

model.add(Conv1D(filters = num_filter[0],
                        kernel_size = length_filter[0],
                        padding = 'same',
                        activation = 'relu',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        input_shape = (echos, depth)))
model.add(BatchNormalization(axis=-1, momentum=0.99))

model.add(MaxPooling1D(pool_size=pool_length[0]))

model.add(Dropout(0.25))


model.add(Conv1D(filters = num_filter[1],
                        kernel_size = length_filter[1],
                        padding = 'same',
                        activation = 'relu',
                        kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization(axis=-1, momentum=0.99))

model.add(MaxPooling1D(pool_size=pool_length[1]))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(3, activation='softmax')) # Regular Neural Net hidden layer

# Compile model
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5, momentum=.6, decay=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# Fit model on training data
model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=1,
          validation_data=(X_test, y_test))

# Evaluate model on test data
Train_Accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
Validation_Accuracy = model.evaluate(X_test, y_test, verbose=0)[1]




