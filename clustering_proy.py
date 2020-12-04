####Clustering (Agrupacion de filas)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


print("")
print("=====================================")
print("          CLUSTERING                 ")
print("=====================================")

dataproy = pd.read_csv(r"C:\Users\julen\Desktop\UNIVERSIDAD\4A\SCCBD (Smart Cities Ciberseguridad y Big Data)\Bigdata\EntregableS5_Julen\Household energy bill data.csv") 

print(dataproy.info())


#---------------------------------------------------------------------------------------------------------------

####CLUSTERING
num_room = dataproy.iloc[:,0]
num_people = dataproy.iloc[:,1]
is_urban = dataproy.iloc[:,8]
amount_paid =dataproy.iloc[:,9]
ave_month =dataproy.iloc[:,6]
house_area =dataproy.iloc[:,2]
# Visualize the data (only petal_length and petal_width)
plt.scatter(house_area,amount_paid, label='True Position') 
plt.xlabel("house area")
plt.ylabel("maount paid")
plt.show()
# plt.scatter(ave_month,amount_paid, label='True Position') 
# plt.xlabel("ave monthly income")
# plt.ylabel("maount paid")
# plt.show()

#Create Clusters

#---------------------------------------------------------------------------------------------------------------
####DBSCAN
clustering = DBSCAN(eps=3, min_samples=2).fit(dataproy)
clustering.labels_

#---------------------------------------------------------------------------------------------------------------
####JERARQUICO
#def plot_dendrogram(model, **kwargs)

# # setting distance_threshold=0 ensures we compute the full tree.
# model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

# model = model.fit(dataproy)
# plt.title('Hierarchical Clustering Dendrogram')
# # plot the top three levels of the dendrogram
# #plot_dendrogram(model, truncate_mode='level', p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()
#---------------------------------------------------------------------------------------------------------------
####KMEANS

# modelKmeans = KMeans(n_clusters=2, init = 'k-means++')
# modelKmeans.fit(dataproy)
# print(modelKmeans.cluster_centers_) 
# print(modelKmeans.labels_) 

# # Visualize how the data has been clustered (only petal_length and petal_width)
# plt.scatter(is_urban,amount_paid, c=modelKmeans.labels_, cmap='rainbow')  
# # plot the centroid coordinates of each cluster
# plt.scatter(modelKmeans.cluster_centers_[:,8], modelKmeans.cluster_centers_[:,9], color='black')
# plt.xlabel("is urban")
# plt.ylabel("amount paid")
# plt.show()


