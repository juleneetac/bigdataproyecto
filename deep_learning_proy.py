import pandas as pd
import numpy as np
# conda install -c conda-forge tensorflow

from sklearn.model_selection import train_test_split

from sklearn import datasets
from tensorflow import keras
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
# import sys
# print(sys.version)



# 1) import data
dataproy = pd.read_csv(r"C:\Users\julen\Desktop\UNIVERSIDAD\4A\SCCBD (Smart Cities Ciberseguridad y Big Data)\Bigdata\EntregableS5_Julen\Household energy bill data.csv") 

print(dataproy.info())

#---------------------------------------------------------------------------------------------------------------

###NOISE
mu, sigma = 0, 0.1 
# creating a noise 
noise = np.random.normal(mu, sigma, [1000,1]) 
print(noise)
dataproy['noise'] = noise

#---------------------------------------------------------------------------------------------------------------

####RANGOS
# Binning numerical columns  //dividimos en 10 grupos dependiendo del amouint paid
#                            // creamos una nueva columna con estos grupos
# Using Pandas               //reparte en 10 grupes del mismo tamaño
dataproy['Cat_amount_paid'] = pd.qcut(dataproy['amount_paid'], q=10, labels=False )

#---------------------------------------------------------------------------------------------------------------

####ATRIBUTE SELECTION 

#Quitamos la columna del amount_paid
dataproy2 = dataproy.drop(['amount_paid', 'Cat_amount_paid'],axis=1)
#dataproy2 = dataproy.drop(['amount_paid'],axis=1)
print(dataproy2.info())

# Split in train and test datasets
# 2D Attributes
# 2) prepare inputs
X = dataproy2
# 3) prepare outputs: a binary class matrix
y = keras.utils.to_categorical(dataproy['Cat_amount_paid'], 10) #divido en 10 y cada categotria es una cplumna
#y = dataproy['Cat_amount_paid']
#y = dataproy['amount_paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

#---------------------------------------------------------------------------------------------------------------

####DEEP LEARNING (con ruido)

#script 7_1
# 4a) Create the model
model = Sequential()
#model = keras.models.Sequential()
model.add(Dense(20, input_dim=10, activation='relu'))  #las capas del medio mas grandes que el input
model.add(Dense(20, activation='relu'))    #esto es una capa mas
model.add(Dense(20, activation='relu'))     #esto es una capa mas
model.add(Dense(10, activation='softmax'))

# 4b) Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4c) Fit the model
model.fit(X, y, epochs=300, batch_size=15, verbose=1)

# 4d) Evaluate the model
score = model.evaluate(X, y, batch_size=15)
print(score)
score




####DEEP LEARNING (sin ruido)

#script 7_1
# 4a) Create the model
model = Sequential()
#model = keras.models.Sequential()
model.add(Dense(18, input_dim=9, activation='relu'))  #las capas del medio mas grandes que el input
model.add(Dense(18, activation='relu'))    #esto es una capa mas
model.add(Dense(18, activation='relu'))     #esto es una capa mas
model.add(Dense(10, activation='softmax'))

# 4b) Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4c) Fit the model
model.fit(X, y, epochs=300, batch_size=15, verbose=1)

# 4d) Evaluate the model
score = model.evaluate(X, y, batch_size=15)
print(score)
score



# #script 7_2
# # 4a) Create the model
# model = Sequential()
# model.add(Dense(18, input_dim=9, activation='relu'))  #las capas del medio mas grandes que el input
# model.add(Dense(10, activation='softmax'))

# print('1.......')
# model.compile(
#         loss = 'categorical_crossentropy',
#         optimizer = 'rmsprop', 
#         metrics = ['accuracy'])
# print('2.......')
# ### FIT THE MODEL
# history = model.fit(X,y, epochs=50, batch_size=15, verbose=1)

# print('3.......')
# ### PREDICT
# Y_pred = model.predict(
#         X, 
#         300*10 // 15+1)

# print('4.......')
# ### EVALUATION
# y_pred = np.argmax(Y_pred, axis=1) 
# print('Matriz de confusión')
# #print(confusion_matrix(X.classes, y_pred))

# plt.figure(figsize=[8, 6])                                                      #perdidas
# plt.plot(history.history['loss'], 'r', linewidth=3.0) 
# plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
# plt.legend(['Pérdidas de entrenamiento', 'Pérdidas de validación'], fontsize=24)
# plt.xlabel('Epocas ', fontsize=22)
# plt.ylabel('Pérdidas', fontsize=22)
# plt.ylim(0,7)
# plt.title('Curvas de pérdidas', fontsize=22) 
# plt.show()
 
# plt.figure(figsize=[8, 6])                                                    #precision
# plt.plot(history.history['accuracy'], 'r', linewidth=3.0) 
# plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
# plt.legend(['Precisión de entrenamiento', 'Precisión de validación'], fontsize=24)
# plt.xlabel('Epocas ', fontsize=22)
# plt.ylabel('Precisión', fontsize=22) 
# plt.ylim(0,1)
# plt.title('Curvas de precisión', fontsize=22)
# plt.show()



