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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
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

###############################################################################
# Organize
###############################################################################
ratio = 0.75
numclasses = 3

# Make sizes equal
min_sample_size = np.min([np.shape(WAT)[2], np.shape(BAT)[2], np.shape(MUS)[2]])
training_size_per_class = int(np.round(min_sample_size * ratio))
testing_size_per_class = int(min_sample_size - training_size_per_class)

y_train = np.zeros((training_size_per_class*3, 1))
for i in range(numclasses):
    y_train[(training_size_per_class*i):(training_size_per_class+(training_size_per_class*i)),0] = i

y_validation = np.zeros((testing_size_per_class*3, 1))
for j in range(numclasses):
    y_validation[(testing_size_per_class*j):(testing_size_per_class+(testing_size_per_class*j)), 0] = j

# Make classes equal
WAT = WAT[:, :, permutation(np.shape(WAT)[2]), :]
BAT = BAT[:, :, permutation(np.shape(BAT)[2]), :]
MUS = MUS[:, :, permutation(np.shape(MUS)[2]), :]


X_train = np.concatenate((
        WAT[:, :, 0:training_size_per_class, :], 
        BAT[:, :, 0:training_size_per_class, :], 
        MUS[:, :, 0:training_size_per_class, :]), axis = 2)

X_validation  = np.concatenate((
        WAT[:, :, training_size_per_class:min_sample_size, :], 
        BAT[:, :, training_size_per_class:min_sample_size, :], 
        MUS[:, :, training_size_per_class:min_sample_size, :]), axis = 2)

X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')

X_train = np.nan_to_num(X_train)
X_validation = np.nan_to_num(X_validation)

X_train = X_train[16, 16, :, :]
X_validation = X_validation[16, 16, :, :]


# Standardize training and testing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_validation)

## one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_validation)

################################################################################
##        Select Subset
################################################################################
subset = range(9)
#subset = [8]


X_train = X_train[:, subset]
X_test = X_test[:, subset]

###############################################################################
##         Create the model
###############################################################################

# Signal Parameters
echos = np.shape(X_train)[1]   # Total amount of time steps
depth = 1  # Real and Imag have been combined
weight_decay = 0.00001#0.00001

num_filter = [32, 16, 16]
length_filter = [4, 4, 4]
pool_length = [2,2,2]

# CNN NETWORK
model = Sequential() # Initializes model to sequientially add layers
model.add(Dense(60, input_shape=(echos,), activation='relu'))



model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.25))


model.add(Dense(3, activation='softmax')) # Regular Neural Net hidden layer

# Compile model
#model.compile(loss='categorical_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])
sgd = optimizers.SGD(lr=0.1, clipvalue=0.5, momentum=.6, decay=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# Fit model on training data
model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Evaluate model on test data
Train_Accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
Validation_Accuracy = model.evaluate(X_test, y_test, verbose=0)[1]


Prediction = model.predict(X_test)

print()
print('Train: ' + str(round(Train_Accuracy * 100, 2)))
print('Validation: ' + str(round(Validation_Accuracy * 100, 2)))

###############################################################################
# Confusion Matrix
###############################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

############# Training Confusion Matrix #######################################
class_names = ["WAT", "BAT", "MUS"]
Training_Prediction = np.argmax(model.predict(X_train), axis=1)
Training_Target = np.argmax(y_train, axis=1)
                            
# Compute confusion matrix
cnf_matrix1 = confusion_matrix(Training_Target, Training_Prediction)
total = np.sum(cnf_matrix1, axis=1)
train_confusion = (np.round(np.vstack((cnf_matrix1[0, :] / total[0], cnf_matrix1[1, :] / total[1], cnf_matrix1[2, :] / total[2])),decimals=3))*100


np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_confusion, classes=class_names, normalize=False,
                      title='Normalized confusion matrix')

plt.show()
                   
############# Validation Confusion Matrix #####################################
class_names = ["WAT", "BAT", "MUS"]
Validation_Prediction = np.argmax(model.predict(X_train), axis=1)
Validation_Target = np.argmax(y_train, axis=1)
                              
# Compute confusion matrix
cnf_matrix2 = confusion_matrix(Validation_Target, Validation_Prediction)
total = np.sum(cnf_matrix1, axis=1)

validation_confusion = (np.round(np.vstack((cnf_matrix2[0, :] / total[0], cnf_matrix2[1, :] / total[1], cnf_matrix2[2, :] / total[2])),decimals=3))*100

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(validation_confusion, classes=class_names, normalize=False,
                      title='Normalized confusion matrix')

plt.show()