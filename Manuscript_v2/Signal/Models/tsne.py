import scipy.io as sio
from sklearn.manifold import TSNE
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

###############################################################################
# Define
###############################################################################
def LoadStackPeak(fileloc, name):
    '''Loads Matlab file and stacks data in proper format'''
    
    file = sio.loadmat(str(fileloc))
    PeakData = file[str(name)]
    return PeakData


###############################################################################
# Load Data
###############################################################################
WAT = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_WAT.csv', delimiter=',')
BAT = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_BAT.csv', delimiter=',')
MUS = np.loadtxt('/home/brandon/Desktop/Signal_Manuscript/Dataset/Reduced/1D/1D_Complex_Signal_Dataset_Reduced_MUS.csv', delimiter=',')

X_data = np.concatenate((WAT, BAT, MUS), axis = 0)

scaler = preprocessing.StandardScaler().fit(X_data)
X_data = scaler.transform(X_data)
# Create class
WAT_class = np.zeros((WAT.shape[0],1), dtype=int)
BAT_class = np.zeros((BAT.shape[0],1), dtype=int)+1
MUS_class = np.zeros((MUS.shape[0],1), dtype=int)+2

target = np.concatenate((WAT_class, BAT_class, MUS_class))

# Run TSNE
X_tsne = TSNE(perplexity=5.0, learning_rate=1000).fit_transform(X_data)

#Plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target, alpha=0.5)
plt.show()

###############################################################################
# Load Data
###############################################################################
WAT_Peak = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/WAT_dataset.mat', 'WAT'))
BAT_Peak = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/BAT_dataset.mat', 'BAT'))
MUS_Peak = np.asarray(LoadStackPeak('/home/brandon/Desktop/Extended_Peak_Dataset/Dataset/2D_Dataset/MUS_dataset.mat', 'MUS'))

WAT_Peak = WAT_Peak[16, 16, :, :]
BAT_Peak = BAT_Peak[16, 16, :, :]
MUS_Peak = MUS_Peak[16, 16, :, :]


X_data2 = np.concatenate((WAT_Peak, BAT_Peak, MUS_Peak), axis = 0)

scaler2 = preprocessing.StandardScaler().fit(X_data2)
X_data2 = scaler2.transform(X_data2)

# Create class
WAT_class_Peak = np.zeros((WAT_Peak.shape[0],1), dtype=int)
BAT_class_Peak = np.zeros((BAT_Peak.shape[0],1), dtype=int)+1
MUS_class_Peak = np.zeros((MUS_Peak.shape[0],1), dtype=int)+2

target_Peak = np.concatenate((WAT_class_Peak, BAT_class_Peak, MUS_class_Peak))

# Run TSNE
X_tsne_Peak = TSNE(perplexity=100.0, learning_rate=100).fit_transform(X_data2)

#Plot
plt.scatter(X_tsne_Peak[:, 0], X_tsne_Peak[:, 1], c=target_Peak, alpha=0.5)
plt.show()