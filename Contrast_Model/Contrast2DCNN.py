import scipy.io as sio
import numpy as np
from sklearn import preprocessing

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras import regularizers
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers

###############################################################################
# Define
###############################################################################
def LoadStackPeak(fileloc):
    '''Loads Matlab file and stacks data in proper format'''
    
    file = sio.loadmat(str(fileloc))
    PeakData = file['dataset']
    return PeakData

def Standardize(dataset):
    """ Dataset format (Samples, Row, Col, Depth) """
    z_score = np.zeros(np.shape(dataset))
    z_score[:,:,:] = (dataset[:,:,:] - dataset[:,:,:].mean()) / dataset[:,:,:].std()
    
    return z_score
###############################################################################
# Load Data
###############################################################################


# August Surface Coil #

# Mouse 1
Surface_Mouse_1_IGWAT_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_1_pre_igwat_mus.mat')
Surface_Mouse_1_IGWAT_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_1_pre_igwat_wat.mat')
Surface_Mouse_1_INTER_BAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_1_pre_intercap_bat.mat')
Surface_Mouse_1_INTER_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_1_pre_intercap_mus.mat')
Surface_Mouse_1_INTER_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_1_pre_intercap_wat.mat')

# Mouse 2
Surface_Mouse_2_IGWAT_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_2_pre_igwat_mus.mat')
Surface_Mouse_2_IGWAT_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_2_pre_igwat_wat.mat')
Surface_Mouse_2_INTER_BAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_2_pre_intercap_bat.mat')
Surface_Mouse_2_INTER_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_2_pre_intercap_mus.mat')
Surface_Mouse_2_INTER_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'aug_mouse_2_pre_intercap_wat.mat')

# February Volume Coil #

# Mouse 1
Volume_Mouse_1_IGWAT_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_1_pre_igwat_mus.mat')
Volume_Mouse_1_IGWAT_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_1_pre_igwat_wat.mat')
Volume_Mouse_1_INTER_BAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_1_pre_intercap_bat.mat')
Volume_Mouse_1_INTER_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_1_pre_intercap_mus.mat')
Volume_Mouse_1_INTER_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_1_pre_intercap_wat.mat')

# Mouse 3
Volume_Mouse_3_IGWAT_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_3_pre_igwat_mus.mat')
Volume_Mouse_3_IGWAT_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_3_pre_igwat_wat.mat')

# Mouse 4
Volume_Mouse_4_INTER_BAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_4_pre_intercap_bat.mat')
Volume_Mouse_4_INTER_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_4_pre_intercap_mus.mat')
Volume_Mouse_4_INTER_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_4_pre_intercap_wat.mat')

# Mouse 5
Volume_Mouse_5_INTER_BAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_5_pre_intercap_bat.mat')
Volume_Mouse_5_INTER_MUS = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_5_pre_intercap_mus.mat')
Volume_Mouse_5_INTER_WAT = LoadStackPeak('/home/brandon/Desktop/PeakDataProject/Datasets/Peak_Data/2D/'
                   'feb_mouse_5_pre_intercap_wat.mat')

###############################################################################
# Standardize and Organize
###############################################################################

WAT_training = np.concatenate((Surface_Mouse_1_IGWAT_WAT, Surface_Mouse_1_INTER_WAT, 
                                       Volume_Mouse_1_IGWAT_WAT, Volume_Mouse_1_INTER_WAT, 
                                       Volume_Mouse_3_IGWAT_WAT, Volume_Mouse_4_INTER_WAT, 
                                       Volume_Mouse_5_INTER_WAT), axis=2)
WAT_validation = np.concatenate((Surface_Mouse_2_IGWAT_WAT, Surface_Mouse_2_INTER_WAT), axis=2)

BAT_training = np.concatenate((Surface_Mouse_1_INTER_BAT, Volume_Mouse_1_INTER_BAT, 
                                       Volume_Mouse_4_INTER_BAT, Volume_Mouse_5_INTER_BAT), axis=2)
BAT_validation = Surface_Mouse_2_INTER_BAT

MUS_training = np.concatenate((Surface_Mouse_1_IGWAT_MUS, Surface_Mouse_1_INTER_MUS, 
                                       Volume_Mouse_1_IGWAT_MUS, Volume_Mouse_1_INTER_MUS, 
                                       Volume_Mouse_3_IGWAT_MUS, Volume_Mouse_4_INTER_MUS, 
                                       Volume_Mouse_5_INTER_MUS), axis=2)
MUS_validation = np.concatenate((Surface_Mouse_2_IGWAT_MUS, Surface_Mouse_2_INTER_MUS), axis=2)

WAT_training = WAT_training[:,:,0:np.shape(MUS_training)[2],:]
BAT_training = BAT_training[:,:,0:np.shape(MUS_training)[2],:]


###############################################################################
# Create class
###############################################################################
WAT_train_class = np.zeros((WAT_training.shape[2],1), dtype=int)
WAT_validation_class = np.zeros((WAT_validation.shape[2],1), dtype=int)

BAT_train_class = np.zeros((BAT_training.shape[2],1), dtype=int)+1
BAT_validation_class = np.zeros((BAT_validation.shape[2],1), dtype=int)+1

MUS_train_class = np.zeros((MUS_training.shape[2],1), dtype=int)+2
MUS_validation_class = np.zeros((MUS_validation.shape[2],1), dtype=int)+2

X_train = np.concatenate((WAT_training, BAT_training, MUS_training), axis=2)
y_train = np.concatenate((WAT_train_class, BAT_train_class, MUS_train_class))
y_train = np.ravel(y_train)

X_validation= np.concatenate((WAT_validation, BAT_validation, 
                              MUS_validation), axis=2)
y_validation= np.concatenate((WAT_validation_class, BAT_validation_class, 
                              MUS_validation_class))
y_validation= np.ravel(y_validation)

X_train = X_train.astype('float32')
X_validation= X_validation.astype('float32')

X_train = np.einsum('ijkl -> kijl', X_train)
X_validation = np.einsum('ijkl -> kijl', X_validation)

X_train = X_train[:,:,:,7]
X_validation = X_validation[:,:,:,7]

X_train = np.expand_dims(X_train, axis=3)
X_validation = np.expand_dims(X_validation, axis=3)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_validation.shape[1]

# Standardize Data
X_train = Standardize(X_train)
X_validation = Standardize(X_validation)

###############################################################################
#        Create the model
###############################################################################


input_shape = (33,33,1)
weight_decay = 0.0001

num_filter = [64,32,32]
length_filter = [(6, 6),(4,4),(3,3)]


model = Sequential()

model.add(Conv2D(filters = num_filter[0],
                 kernel_size = length_filter[0],
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=input_shape))

model.add(Conv2D(filters = num_filter[1],
                 kernel_size = length_filter[1],
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(4,4),
                       padding='valid'))

#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(Dropout(0.25))


model.add(Conv2D(filters = num_filter[1],
                 kernel_size = length_filter[1],
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(4,4),
                       padding='valid'))

#model.add(BatchNormalization(axis=-1, momentum=0.99))
model.add(Dropout(0.25))

#model.add(Conv2D(filters = num_filter[2],
#                 kernel_size = length_filter[2],
#                 strides=(1,1),
#                 padding='same',
#                 activation='relu',
#                 kernel_regularizer=regularizers.l2(weight_decay),
#                 input_shape=input_shape))
#
#model.add(MaxPooling2D(pool_size=(2,2),
#                       padding='valid'))
#
#model.add(BatchNormalization(axis=-1, momentum=0.99))
#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64, activation='relu'))



model.add(Dropout(0.25))

model.add(Dense(3, activation='softmax'))

# Compile model
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5, momentum=.5, decay=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
metrics=['accuracy'])

# Fit model on training data
model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=1,
          validation_data=(X_validation, y_validation))


model.save_weights('2DCNN_first.h5') 
# Evaluate model on test data
Train_Accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
Validation_Accuracy = model.evaluate(X_validation, y_validation, verbose=0)[1]




