# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

import pandas as pd
import numpy as np

###############################################################################
## Carregar o dataset
###############################################################################
DATASET_FILE = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
print ('Dataset:', DATASET_FILE)
df = pd.read_csv ('../MachineLearningCVE/' + DATASET_FILE, low_memory = False)
input ('Dataset carregado.')

###############################################################################
## Tratar valores faltantes (NaN, Infinity), atualmente substituidos por  0
###############################################################################
#print (df.isnull ().any ())
print ('Existem valores faltantes (NaN) no dataframe:', df.isnull ().values.any ())
colunasNaN = [i for i in df.columns if df [i].isnull ().any ()]
print ('Colunas com dados faltantes (NaN):', colunasNaN)
df.replace ('Infinity', np.nan, inplace = True) ## Remover infinitos tambem
df.replace (np.inf, np.nan, inplace = True) ## Remover infinitos tambem
# df.replace (np.nan, 0, inplace = True)
## Ao inves de substituir os valores por 0, usaremos um simpleImputer
## da biblioteca scikit para colocar valores medios/medianos no lugar
print ('Existem valores faltantes (NaN) no dataframe:', df.isnull ().values.any ())
colunasNaN = [i for i in df.columns if df [i].isnull ().any ()]
print ('Colunas com dados faltantes (NaN):', colunasNaN)
input ('Dataset tratado.')


###############################################################################
## Exibir informacoes sobre o dataset
###############################################################################
print ('Tipo preliminar do dataframe:', type (df), '\n')
print ('Formato do dataframe (linhas, colunas):', df.shape, '\n')
print ('Primeiras 5 linhas do dataframe:\n', df [:5], '\n')
#print ('Atributos do dataframe:\n', df.keys ())
df.info (verbose = True)
#print (df.describe ())
input ('Dataset analisado.')
## Ate aqui trabalhavamos com um dataframe pandas

###############################################################################
## Codificar rotulos das amostras
###############################################################################
print ('Rotulos das amostras antes da conversao:', df [' Label'].unique ())
df [' Label'] = df [' Label'].replace ('BENIGN', 0)
df [' Label'] = df [' Label'].replace ('DDoS', 1)
print ('Rotulos das amostras apos a conversao:', df [' Label'].unique ())
print ('Dataset apos conversao dos rotulos:')
df.info (verbose = True)
input ('Dataset codificado.')

###############################################################################
## Converter dataframe para arrays numpy
###############################################################################
X = df.iloc [:, :-1].values
y = df.iloc [:, -1].values
### K: Substituir valores faltantes pela mediana do atributo.
from sklearn.impute import SimpleImputer
myImputer = SimpleImputer (missing_values = np.nan, strategy = 'median')
myImputer = myImputer.fit (X [:, :-1])
X [:, :-1] = myImputer.transform (X [:, :-1])
print ('Tipo do dataset (atributos):', type (X))
print ('Tipo do dataset (rotulos):', type (y))
print ('Dataset contem valores NaN:', np.any (np.isnan (X)))
print ('Dataset contem apenas valores finitos:', np.all (np.isfinite (X)))
input ('Dataset convertido.')

###############################################################################
## Separar os conjuntos de treino e teste
###############################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 1/3, random_state = 0)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)
#print ('X_test:', X_test)
#print ('y_test:', y_test)
input ('Dataset dividido.')

###############################################################################
## Instanciar o modelo de aprendizado
###############################################################################
from keras.models import Sequential
from keras.layers import Dense, Dropout
numberOfClasses = 2 # Classificacao binaria: Benign, DoS
batchSize = 128
numberOfEpochs = 20
model = Sequential ()
model.add (Dense (units = 512, activation = 'relu', input_shape= (78, )))
#model.add (Dropout (0.2))
model.add (Dense (512, activation = 'relu'))
#model.add (Dropout (0.2))
model.add (Dense (256, activation = 'relu'))
#model.add (Dropout (0.2))
model.add (Dense (numberOfClasses, activation = 'softmax'))
print ('Model summary:')
model.summary ()

###############################################################################
## Compilar a rede
###############################################################################
from keras.optimizers import RMSprop
from keras.optimizers import Adam
model.compile (loss = 'sparse_categorical_crossentropy',
               optimizer = Adam (lr=0.001),
               #optimizer = RMSprop (),
               metrics= ['accuracy'])

###############################################################################
## Treinar (fit) a rede
###############################################################################
history = model.fit (X_train, y_train,
                     batch_size = batchSize,
                     epochs = numberOfEpochs,
                     verbose = 1,
                     validation_data = (X_test, y_test))

###############################################################################
## Avaliar os resultados
###############################################################################
scoreArray = model.evaluate (X_test, y_test, verbose = 0)
print ('Test loss:', scoreArray [0])
print ('Test accuracy:', scoreArray [1])
