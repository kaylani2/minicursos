# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Vinayakumar probably altered the original dataset for multiclass classification.
### K: TODO:
  # Remove categorical features for fitting
  # Group classes into superclasses: normal, dos, probe, u2r and r2l

from __future__ import print_function
#from sklearn.cross_validation import train_test_split ## K: deprecated
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed (1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, mean_squared_error, mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

TRAIN_FILE = 'newTrain.txt'
TEST_FILE  = 'newTest.txt'

#from scipy.io import arff
#import pandas as pd
#data = arff.loadarff ('test.arff')
#df = pd.DataFrame (data[0])
#df.head ()

###############################################################################
## Carregar o dataset
###############################################################################
trainFrame = pd.read_csv (TRAIN_FILE)#, header = None)
testFrame = pd.read_csv (TEST_FILE)#, header = None)
df = pd.concat ([trainFrame, testFrame], ignore_index = True)

print ('____________________________________________________________________________________________________________')
###############################################################################
## Exibir informacoes sobre o dataset
###############################################################################
print ('Tipo preliminar do dataframe:', type (df), '\n')
print ('Formato do dataframe (linhas, colunas):', df.shape, '\n')
print ('Primeiras 5 linhas do dataframe:\n', df [:5], '\n')
#print ('Atributos do dataframe:\n', df.keys ())
print ('Informacao:')
df.info (verbose = True)
print ('Descricao:')
#print (df.describe ())
print ('Tipos de ataque:', df ['class'].unique ())
print ('Quantidade de ataques:', len (df ['class'].unique ()))
print ('Graus de severidade de ataques:', df ['severity'].unique ())
#input ('Dataset analisado.')
## Ate aqui trabalhavamos com um dataframe pandas

###############################################################################
## Converter dataframe para arrays numpy
###############################################################################
X = df.iloc [:, :-2].values
y = df.iloc [:, -2].values

###############################################################################
## Tratar atributos categóricos
###############################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder ()
X[:, 1] = labelencoder_X.fit_transform (X[:, 1])
X[:, 2] = labelencoder_X.fit_transform (X[:, 2])
X[:, 3] = labelencoder_X.fit_transform (X[:, 3])
labelencoder_y = LabelEncoder ()
y = labelencoder_y.fit_transform (y)
print ('Informacao:')
df.info (verbose = True)

###############################################################################
## Separar os conjuntos de treino e teste
###############################################################################
from sklearn.model_selection import train_test_split
trainX, testT, y_train, y_test = train_test_split (X, y, test_size = 1/3, random_state = 0)
#print ('X_train shape:', X_train.shape)
#print ('y_train shape:', y_train.shape)
#print ('X_test shape:', X_test.shape)
#print ('y_test shape:', y_test.shape)
#print ('X_test:', X_test)
#print ('y_test:', y_test)

#X = trainData.iloc[:, 1:42]
#Y = trainData.iloc[:, 0]
#C = testData.iloc[:, 0]
#T = testData.iloc[:, 1:42]

#scaler = Normalizer ().fit (X)
#trainX = scaler.transform (X)
## summarize transformed data
#np.set_printoptions (precision = 3)
##print (trainX[0:5, :])
#
#scaler = Normalizer ().fit (T)
#testT = scaler.transform (T)
## summarize transformed data
#np.set_printoptions (precision = 3)
##print (testT[0:5, :])
#
#
#y_train1 = np.array (Y)
#y_test1 = np.array (C)
#
#y_train= to_categorical (y_train1)
#y_test= to_categorical (y_test1)



# reshape input to be [samples, time steps, features]
X_train = np.reshape (trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape (testT, (testT.shape[0], 1, testT.shape[1]))
#y_train = np.reshape (y_train, (y_train.shape[0], 1, y_train.shape[1]))
#y_test = np.reshape (y_test, (y_test.shape[0], 1, y_test.shape[1]))


batch_size = 32

model = Sequential ()
model.add (SimpleRNN (4, input_shape = (X_train.shape [1:])))  # try using a GRU instead, for fun
model.add (Dropout (0.1))
model.add (Dense (40))
model.add (Activation ('softmax'))

model.compile (loss = 'sparse_categorical_crossentropy',
               optimizer = 'adam',
               metrics = ['accuracy'])
#checkpointer = callbacks.ModelCheckpoint (filepath = "kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5",
#                                          verbose = 1, save_best_only = True, monitor = 'val_acc', mode = 'max')
#csv_logger = CSVLogger ('training_set_iranalysis.csv', separator = ', ', append = False)
model.fit (X_train, y_train,
           batch_size = batch_size,
           epochs = 3,
           validation_data = (X_test, y_test))#, callbacks = [checkpointer, csv_logger])

model.save ('./model.hdf5')
#model.save ("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")

loss, accuracy = model.evaluate (X_test, y_test)
print ("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes (X_test)
#np.savetxt ('kddresults/lstm1layer/lstm1predicted.txt', np.transpose ([y_test, y_pred]), fmt = '%01d')
