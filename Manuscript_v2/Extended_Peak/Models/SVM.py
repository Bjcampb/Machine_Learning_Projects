import scipy.io as sio
import numpy as np
from numpy.random import permutation
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing

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

y_train = np.ravel(y_train)
y_validation = np.ravel(y_validation)
################################################################################
##        Select Subset
################################################################################
subset = range(9)
#subset = [7]


X_train = X_train[:, subset]
X_test = X_test[:, subset]

#############################################################################
## Create SVM
###############################################################################
clf = SVC(C=35.0, gamma=.01, probability=True)
#clf = LinearSVC()
clf.fit(X_train, y_train)
Train_Accuracy = clf.fit(X_train, y_train).score(X_train, y_train)
test_prediction = clf.predict(X_test)
coefs = clf.dual_coef_

y_validation = np.ravel(y_validation)
Test_accuracy = test_prediction == y_validation
Test_accuracy = sum(Test_accuracy)
Test_accuracy = Test_accuracy / np.shape(y_validation)[0]
#Validation_Accuracy = clf.predict_proba(X_test)

print()
print('Train: ' + str(round(Train_Accuracy * 100, 2)))
print('Validation: ' + str(round(Test_accuracy * 100, 2)))

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
train_confusion = np.round(np.vstack((cnf_matrix1[0, :] / total[0], cnf_matrix1[1, :] / total[1], cnf_matrix1[2, :] / total[2])),decimals=3)


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

validation_confusion = np.round(np.vstack((cnf_matrix2[0, :] / total[0], cnf_matrix2[1, :] / total[1], cnf_matrix2[2, :] / total[2])),decimals=3)

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(validation_confusion, classes=class_names, normalize=False,
                      title='Normalized confusion matrix')

plt.show()