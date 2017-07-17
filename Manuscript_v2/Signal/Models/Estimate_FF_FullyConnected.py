import scipy.io as sio
import numpy as np
from numpy.random import permutation
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.utils import np_utils
from keras import optimizers

from sklearn import linear_model
import matplotlib.pyplot as plt
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
WAT = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_WAT.csv', delimiter=',')
BAT = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_BAT.csv', delimiter=',')
MUS = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_MUS.csv', delimiter=',')

WAT_FF = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/WAT_dataset.mat', 'WAT'))
BAT_FF = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/BAT_dataset.mat', 'BAT'))
MUS_FF = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/MUS_dataset.mat', 'MUS'))

WAT_FF = WAT_FF[16,16,:,8]
BAT_FF = BAT_FF[16,16,:,8]
MUS_FF = MUS_FF[16,16,:,8]

###############################################################################
# Organize
###############################################################################
ratio = 0.75
numclasses = 3

# Make sizes equal
min_sample_size = np.min([np.shape(WAT)[0], np.shape(BAT)[0], np.shape(MUS)[0]])
training_size_per_class = int(np.round(min_sample_size * ratio))
testing_size_per_class = int(min_sample_size - training_size_per_class)

#y_train = np.zeros((training_size_per_class*3, 1))
#for i in range(numclasses):
#    y_train[(training_size_per_class*i):(training_size_per_class+(training_size_per_class*i)),0] = i
#
#y_validation = np.zeros((testing_size_per_class*3, 1))
#for j in range(numclasses):
#    y_validation[(testing_size_per_class*j):(testing_size_per_class+(testing_size_per_class*j)), 0] = j

# Make classes equal
WAT = WAT[permutation(np.shape(WAT)[0]), :]
BAT = BAT[permutation(np.shape(BAT)[0]), :]
MUS = MUS[permutation(np.shape(MUS)[0]), :]

WAT_FF = WAT_FF[permutation(np.shape(WAT)[0])]
BAT_FF = BAT_FF[permutation(np.shape(BAT)[0])]
MUS_FF = MUS_FF[permutation(np.shape(MUS)[0])]

X_train = np.concatenate((
        WAT[0:training_size_per_class, :], 
        BAT[0:training_size_per_class, :], 
        MUS[0:training_size_per_class, :]), axis = 0)

X_validation  = np.concatenate((
        WAT[training_size_per_class:min_sample_size, :], 
        BAT[training_size_per_class:min_sample_size, :], 
        MUS[training_size_per_class:min_sample_size, :]), axis = 0)

X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')

#y_train = X_train[:,24]
#y_test = X_validation[:,24]

y_train = np.concatenate((
        WAT_FF[0:training_size_per_class], 
        BAT_FF[0:training_size_per_class], 
        MUS_FF[0:training_size_per_class]), axis = 0)

y_test  = np.concatenate((
        WAT_FF[training_size_per_class:min_sample_size], 
        BAT_FF[training_size_per_class:min_sample_size], 
        MUS_FF[training_size_per_class:min_sample_size]), axis = 0)


# Standardize training and testing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_validation)
#
#X_train = np.expand_dims(X_train, axis=2)
#X_test = np.expand_dims(X_test, axis=2)
#
## one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_validation)


###############################################################################
##         Create the model
###############################################################################
X_train = X_train[:,0:24]
X_test = X_test[:,0:24]




# CNN NETWORK
model = Sequential() # Initializes model to sequientially add layers
#model.add(Dense(126, input_shape=(24,), activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(1, activation='softmax')) # Regular Neural Net hidden layer
model.add(Dense(64, input_shape=(24,), activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dropout())
model.add(Dense(15))
model.add(Dropout(0.25))
model.add(Dense(1))
# Compile model
#model.compile(loss='mse',
#              optimizer='adam',
#              metrics=['accuracy'])

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


# Fit model on training data
model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=1, validation_data=(X_test, y_test))



Prediction = model.predict(X_test)

from sklearn.metrics import r2_score

R2_score = r2_score(y_test, Prediction)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
Test = np.expand_dims(y_test, axis=1)
regr.fit(Test, Prediction)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(Test) - Prediction) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(Test, Prediction))

# Plot outputs
plt.scatter(Test, Prediction,  color='black')
plt.plot(Test, regr.predict(Test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
