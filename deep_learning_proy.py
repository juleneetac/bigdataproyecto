import pandas as pd
# conda install -c conda-forge tensorflow

from sklearn import datasets
from tensorflow import keras
from tensorflow.keras.layers import Dense
# import sys
# print(sys.version)



# 1) import data
dataproy = pd.read_csv(r"C:\Users\julen\Desktop\UNIVERSIDAD\4A\SCCBD (Smart Cities Ciberseguridad y Big Data)\Bigdata\EntregableS5_Julen\Household energy bill data.csv") 

print(dataproy.info())



#---------------------------------------------------------------------------------------------------------------

####RANGOS
# Binning numerical columns  //dividimos en 10 grupos dependiendo del amouint paid
#                            // creamos una nueva columna con estos grupos
# Using Pandas               //reparte en 10 grupes del mismo tamaño
dataproy['Cat_amount_paid'] = pd.qcut(dataproy['amount_paid'], q=10, labels=False )

#---------------------------------------------------------------------------------------------------------------

####ATRIBUTE SELECTION   (Brute Force)

#Quitamos la columna del amount_paid
dataproy2 = dataproy.drop(['amount_paid', 'Cat_amount_paid'],axis=1)
#dataproy2 = dataproy.drop(['amount_paid'],axis=1)
print(dataproy2.info())

# Split in train and test datasets
# 2D Attributes
# 2) prepare inputs
input_x = dataproy2
# 3) prepare outputs: a binary class matrix
output_y = keras.utils.to_categorical(dataproy['Cat_amount_paid'], 10) #divido en 10 y cada categotria es una cplumna
#y = dataproy['Cat_amount_paid']
#y = dataproy['amount_paid']




# 4a) Create the model
model = keras.models.Sequential()
model.add(Dense(8, input_dim=9, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 4b) Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4c) Fit the model
model.fit(input_x, output_y, epochs=150, batch_size=15, verbose=1)

# 4d) Evaluate the model
score = model.evaluate(input_x, output_y, batch_size=15)
print(score)
score
