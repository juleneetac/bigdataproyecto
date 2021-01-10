import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

# Atribute selection
from sklearn.feature_selection import SelectFromModel  # Regularization (L1 norm)
from sklearn.linear_model import LogisticRegression

#sklearn variaos:
from sklearn.model_selection import train_test_split

#Classifiers:
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#Recomenders:
from sklearn.metrics.pairwise import cosine_similarity
import operator


# Load from a CSV
data = pd.read_csv(r"C:\Users\julen\Desktop\DatasetsPy\UJIIndoorLoc_ID.csv") 

# Clase: categorical to numerical
data['ID'] = data['ID'].astype('category')
data2 = data.loc[:,(data !=100).any()]
#Quitamos la columna del ID
data3 = data2.iloc[:,:-1]

print(data3.info())

#-----------------------------------------------------------------------------------------
######weighted average ratings

#cambiamos los 100 por nan
s3 = data3.replace([100], np.nan)
means = s3.mean()

wavg = means.sort_values(ascending= False)
recommended10 = wavg.head(10)
ax1 = sns.barplot(x = recommended10,  y =recommended10.index ,palette = 'deep')
plt.xlim(-70, -56)
plt.xlabel('potencia', weight = 'bold')
plt.ylabel('WAPxx', weight = 'bold')

print("Recomendamos: ")
print(recommended10)


#----------------------------------------------------------------------------------------
######SABIENDO QUE 100 SON LOS QUE NO LLEGAN COJEMOS LOS QUE MENOS 100 TIENEN Y SERAN LOS QUE MAS SE USAN
solo100 = data3
solo100[solo100 != 100] = 0   #todos los valores que no son 0 los cambiamos por 0
sumasolo100= solo100.sum(axis=0)   #para aqui sumar todas las filas de cada columna (axis = 0 es columnas)
sumasolo100 = sumasolo100.sort_values(ascending=True)
topsuma= sumasolo100[:10]
ax2 = sns.barplot(x = topsuma,  y =topsuma.index ,palette = 'deep')
plt.xlim(1510000, 1590000)
plt.xlabel('Menos 100 (valor no util)', weight = 'bold')
plt.ylabel('WAPxx', weight = 'bold')
print("Los mas usados son:")
print(topsuma)

#----------------------------------------------------------------------------------------
#######SE PODRIA HACER TAMBIEN LAS SALAS QUE MAS AP TIENEN
solo100 = data3
solo100[solo100 != 100] = 0 
sumafilasT = solo100.T    #Hacemos la transpuesta
sumasfilasT= sumafilasT.sum(axis=0)   #suma de columnas
sumasfilasT = sumasfilasT.sort_values(ascending=True)
topsumafilas= sumasfilasT[:10]
topsumafilas.index = topsumafilas.index.astype('category')
ax3 = sns.barplot(x = topsumafilas,  y = topsumafilas.index, palette = 'deep')
plt.xlim(41300, 42000)
plt.xlabel('Menos suma de 100s (valor no util)', weight = 'bold')
plt.ylabel('Salas', weight = 'bold')
print("Las salas que le llegan mas WAPxx son:")
print(topsumafilas)

#----------------------------------------------------------------------------------------
#######COMPARA UNO CON EL RESTO

def similar_zones(zone_id, matrix, k=3):
    # create a df of just the current user    //hemos cambiado el .index por ['ID'] para coger la ID de las zonas
    zone = matrix[matrix['ID'] == zone_id]
    
    # and a df of all other users
    other_zones = matrix[matrix['ID'] != zone_id]
    
    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(zone,other_zones)[0].tolist()
    
    # create list of indices of these users
    indices = other_zones['ID'].tolist()
    
    # create key/values pairs of user index and their similarity
    index_similarity = dict(zip(indices, similarities))
    
    # sort by similarity
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    
    # grab k users off the top
    top_zones_similarities = index_similarity_sorted[:k]
    zones = [u[0] for u in top_zones_similarities]
    
    return zones
    
current_zone = 12106    #ponemos el id de una zona concreta (ojo no el index)
# try it out
similar_zone_indices = similar_zones(current_zone, data2)
print(similar_zone_indices)







#----------------------------------------------------------------------------------------
# s3transform = s3.T
                    
# V = s3transform.iloc[0: , [0]].mean()   #iloc[1:3, 0:3]  //filas ---columnas
# V2 = s3transform.mean() 