import numpy as np # linear algebra
import pandas as pd # data processing


#sklearn variaos:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#Classifiers:
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report

#Mejora del tree classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#Scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#CPlot:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

#utils
from sklearn import utils

# Logistic regresion:
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import mean_absolute_error

#Regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Preprocesing
# from sklearn.preprocessing import KBinsDiscretizer


#Optimizacion
from sklearn.model_selection import GridSearchCV



# Load from a CSV
dataproy = pd.read_csv(r"C:\Users\julen\Desktop\UNIVERSIDAD\4A\SCCBD (Smart Cities Ciberseguridad y Big Data)\Bigdata\EntregableS5_Julen\Household energy bill data.csv") 

#dataproy['amount_paid'] = dataproy['amount_paid'].astype('int64')
# dataproy['amount_paid']=dataproy['amount_paid'].cat.codes
#dataproy['housearea' ] = dataproy['housearea'].astype('int64')
#dataproy['ave_monthly_income' ] = dataproy['ave_monthly_income'].astype('int64')
# Attribute information // me da los tipos de cada columna
print(dataproy.info())



#---------------------------------------------------------------------------------------------------------------
####RANGOS
# Binning numerical columns  //dividimos en 10 grupos dependiendo del amouint paid
#                            // creamos una nueva columna con estos grupos
# Using Pandas               //reparte en 10 grupes del mismo tamaño
dataproy['Cat_amount_paid'] = pd.qcut(dataproy['amount_paid'], q=10, labels=False )
#---------------------------------------------------------------------------------------------------------------

####ATRIBUTE SELECTION

#Quitamos la columna del amount_paid
dataproy2 = dataproy.drop(['amount_paid', 'Cat_amount_paid'],axis=1)  #cambir cuandomusamosm clasificadores normales
#dataproy2 = dataproy.drop(['amount_paid'],axis=1)                    #cambiar cuando usamos regression
print(dataproy2.info())

# Split in train and test datasets
# 2D Attributes
X = dataproy2
y = dataproy['Cat_amount_paid']       #cambir cuando usamos clasificadores normales
#y = dataproy['amount_paid']          #cambiar cuando usamos regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

#---------------------------------------------------------------------------------------------------------------
#### KNN REGRESION   (para rangos)
# print('KNeighborsRegression...')
# for i, weights in enumerate(['uniform', 'distance']):
#     knn_model = neighbors.KNeighborsRegressor(10, weights=weights)
#     knn_model.fit(X_train, y_train)
#     # test prediction
#     y_pred = knn_model.predict(X_test)
#     scores_regr = mean_absolute_error(y_test, y_pred)
#     print(scores_regr)  #dividir esto por el maximo o la media

# #---------------------------------------------------------------------------------------------------------------
# #### Decision Tree Regression with AdaBoost REGRESION   (para rangos)

# Treeregr_1 = DecisionTreeRegressor(criterion='mse', max_depth=10, splitter='random')

# Adaregr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),
#                           n_estimators=500)

# Treeregr_1.fit(X_train, y_train)
# Adaregr_2.fit(X_train, y_train)

# # Predict
# y_pred1 = Treeregr_1.predict(X_test)
# y_pred2 = Adaregr_2.predict(X_test)
# scores_regrTree = mean_absolute_error(y_test, y_pred1)
# scores_regrAda = mean_absolute_error(y_test, y_pred2)

# print('DecisionTreeRegressor...')
# print(scores_regrTree)  #dividir esto por el maximo o la media
# print('AdaBoostRegressor...')
# print(scores_regrAda)  #dividir esto por el maximo o la media


#---------------------------------------------------------------------------------------------------------------

####DECISION TREE
print('DecisionTreeClassifierINICIAL con max_depth=15...')
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=1)
tree_model.fit(X_train, y_train)

# test prediction
y_pred = tree_model.predict(X_test)
print('Accuracy con 10 max_depht: %.2f%%' % (100.0 * tree_model.score(X_test, y_test)))

print('Sin nada: ', classification_report(y_test, y_pred)[-162:])  #classification repor


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


#----------------------------------------------------------------------------------------------------------

# print('RANDOM FOREST...')
# forest = RandomForestClassifier(max_depth=10, 
#                                 n_estimators=100,
#                                 random_state=1)
# forest = forest.fit(X_train, y_train)
# y_test_pred= forest.predict(X_test)
# y_pred_prob = forest.predict_proba(X_test)[:,1]
# print('Accuracy con optimizado: %.2f%%' % (100.0 * forest.score(X_test, y_test)))
# print('Forest: ', classification_report(y_test, y_test_pred)[-162:])




#----------------------------------------------------------------------------------------------------------

# print('BAGGING...')

# bag = BaggingClassifier(base_estimator=tree_model2,
#                         n_estimators=500, 
#                         max_samples=1.0, 
#                         max_features=1.0, 
#                         bootstrap=True, 
#                         bootstrap_features=False, 
#                         n_jobs=1, 
#                         random_state=1)
# bag = bag.fit(X_train, y_train)
# y_test_pred = bag.predict(X_test)
# y_pred_prob = bag.predict_proba(X_test)[:,1]

# print('Accuracy con optimizado: %.2f%%' % (100.0 * bag.score(X_test, y_test)))
# print('Bagging: ', classification_report(y_test, y_test_pred)[-162:])

# #K-FOLD CROSS-VALIDATION
# cv_scores = cross_val_score(bag, X, y, cv=5)
# print("Cross-Validation Accuracy: %0.4f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

#----------------------------------------------------------------------------------------------------------

###BOOSTING
# print('BOOSTING...')
# ada = AdaBoostClassifier(base_estimator=tree_model2,
#                          n_estimators=10, 
#                          learning_rate=0.1,
#                          random_state=1)

# ada = ada.fit(X_train, y_train)
# y_test_pred = ada.predict(X_test)
# y_pred_prob = ada.predict_proba(X_test)
# print('Accuracy con optimizado: %.2f%%' % (100.0 * ada.score(X_test, y_test)))
# print('AdaBoostClassifier: ', classification_report(y_test, y_test_pred)[-162:])

#----------------------------------------------------------------------------------------------------------
# #BUSQUEDA DE DE LOS VALOREWS OPTIMOS deldecision tree
# print('Buscando valores optimos...')
# tree = DecisionTreeClassifier()

# param_grid = {'max_depth': list(range(10, 45)),
#               'criterion': ['entropy', 'gini'],
#               'splitter': ['best', 'random']}

# # Instantiate the search
# tree_cv = GridSearchCV(tree, param_grid, cv = 10)

# # Fit the classifier to the data
# tree_cv.fit(X_train, y_train)

# # Print the tuned parameters and score
# print("Best parameters: {}".format(tree_cv.best_params_)) 
# print("Best score: {}".format(tree_cv.best_score_))

#---------------------------------------------------------------------------------------------------------------

####KNN
# model train metric='euclidean', n_neighbors=1
# print('KNeighborsClassifier...')
# knn_model = KNeighborsClassifier(metric='minkowski', n_neighbors=10)
# knn_model.fit(X_train, y_train)

# # test prediction
# y_pred = knn_model.predict(X_test)
# print('Accuracy: %.2f%%' % (100.0 * knn_model.score(X_test, y_test)))
# print('Sin nada: ', classification_report(y_test, y_pred)[-162:])

#---------------------------------------------------------------------------------------------------------------

# ####SVC
# print('SVC...')
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)


# # test prediction
# y_pred = svm_model.predict(X_test)
# print('Accuracy SVC: %.2f%%' % (100.0 * svm_model.score(X_test, y_test)))
# print(classification_report(y_test, y_pred)[-162:])  

#---------------------------------------------------------------------------------------------------------------
# #####BUSQUEDA DE DE LOS VALOREWS OPTIMOS del SVC  (tarda mucho mucho no se ha hecho alfinal)
# param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
# grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
# grid.fit(X_train,y_train)
    
#---------------------------------------------------------------------------------------------------------------

# ####TODOS LOS CLASSIFIERS
# names = ["Nearest Neighbors", "Decision Tree", "Naive Bayes", "Linear SVM", "Neural Net"]
# classifiers = [
#     KNeighborsClassifier(metric='minkowski', n_neighbors=10),
#     DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=1),
#     GaussianNB(),
#     SVC(kernel="linear"),
#     MLPClassifier(alpha=1, max_iter=1000)]

# #iterate over classifiers
# cv_accuracy = dict()
# for name, clf in zip(names, classifiers):
#     print(name)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print('Accuracy: %.2f%%' % (100.0 * clf.score(X_test, y_test)))
#     #Mostramos las ultimas líneas del classification report, con la media entre clases.
#     print(classification_report(y_test, y_pred)[-162:])

#---------------------------------------------------------------------------------------------------------------

####PLOT
# Correlation matrix
# def plotCorrelationMatrix(df, graphWidth):
#    # filename = df.dataframeName
#     df = df.dropna('columns') # drop columns with NaN
#     df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
#     if df.shape[1] < 2:
#         print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
#         return
#     corr = df.corr()
#     plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
#     corrMat = plt.matshow(corr, fignum = 1)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.gca().xaxis.tick_bottom()
#     plt.colorbar(corrMat)
#     plt.title(f'Correlation Matrix for Household energy bill data', fontsize=15)
#     plt.show()
    
# plotCorrelationMatrix(dataproy, 8)


# # Distribution graphs (histogram/bar graph) of column data
# def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
#     nunique = df.nunique()
#     df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
#     nRow, nCol = df.shape
#     columnNames = list(df)
#     nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
#     plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
#     for i in range(min(nCol, nGraphShown)):
#         plt.subplot(nGraphRow, nGraphPerRow, i + 1)
#         columnDf = df.iloc[:, i]
        
#         if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
#             valueCounts = columnDf.value_counts()
#             valueCounts.plot.bar()
#         else:
#             columnDf.hist()
#         plt.ylabel('counts')
#         plt.xticks(rotation = 90)
#         plt.title(f'{columnNames[i]} (column {i})')
#     plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
#     plt.show()
    
# plotPerColumnDistribution(dataproy, 10, 5)

