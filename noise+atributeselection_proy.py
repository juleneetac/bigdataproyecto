import numpy as np # linear algebra
import pandas as pd # data processing

# Atribute selection
from sklearn.feature_selection import SelectFromModel  # Regularization (L1 norm)
from sklearn.linear_model import LogisticRegression

#sklearn variaos:
from sklearn.model_selection import train_test_split

#Classifiers:
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# Load from a CSV
dataproy = pd.read_csv(r"C:\Users\julen\Desktop\UNIVERSIDAD\4A\SCCBD (Smart Cities Ciberseguridad y Big Data)\Bigdata\EntregableS5_Julen\Household energy bill data.csv") 

# Attribute information // me da los tipos de cada columna
print(dataproy.info())

#---------------------------------------------------------------------------------------------------------------

###NOISE
mu, sigma = 0, 0.1 
# creating a noise 
noise = np.random.normal(mu, sigma, [1000,1]) 
print(noise)
dataproy['noise'] = noise

#---------------------------------------------------------------------------------------------------------------

###RANGOS
# Binning numerical columns  //dividimos en 10 grupos dependiendo del amouint paid
#                             // creamos una nueva columna con estos grupos
# Using Pandas               //reparte en 10 grupes del mismo tamaño
dataproy['Cat_amount_paid'] = pd.qcut(dataproy['amount_paid'], q=10, labels=False )

####ATRIBUTE SELECTION

#Quitamos la columna del amount_paid
dataproy2 = dataproy.drop(['amount_paid', 'Cat_amount_paid'],axis=1)
print(dataproy2.info())


# # Brute force with coorrelation      // no lo usamos porque da correlaciones muyb pequeñas y no es util
# # Unsupervised Features correlation 
# X = dataproy2
# correlated_features = set()
# _correlation_matrix = X.corr(method='spearman')
# for i in range(len(_correlation_matrix.columns)):
#     for j in range(i):
#         if abs(_correlation_matrix.iloc[i, j]) > 0.1:
#             _colname = _correlation_matrix.columns[i]
#             correlated_features.add(_colname)

# print("Unsupervised brute force")
# print("Strong correlated features")
# print(_correlation_matrix)
# print(correlated_features)

# Split in train and test datasets
# 2D Attributes
X = dataproy2
y = dataproy['Cat_amount_paid']       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

# Regularization (L1 norm)
# Using sklearn.feature_selection.SelectFromModel


logistic = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=10000)
sfm = SelectFromModel(estimator=logistic)
sfm.fit(X, y)
feature_importances = pd.DataFrame({'feature':X.columns,'importance':sfm.get_support()})

print("Regularization (L1 norm) Embedded approach")
print(feature_importances)



#----------------------------------------------------------------------------------------------------------
#####ELECCIÓN DEL CLASIFICADOR CON LOS MEJORES VCALORES
# DecisionTreeClassifier
print('DecisionTreeClassifier OPTIMIZADO...')
tree_model2 = DecisionTreeClassifier(criterion='gini', max_depth=10, splitter='random', random_state=1)
tree_model2.fit(X_train, y_train)


# test prediction
y_pred = tree_model2.predict(X_test)
print('Accuracy con optimizado: %.2f%%' % (100.0 * tree_model2.score(X_test, y_test)))
print('Sin nada optimizado: ', classification_report(y_test, y_pred)[-162:])  #classification report// me da el porcentaje
