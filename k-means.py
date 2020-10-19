# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:55:54 2020

@author: dorsaf
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


data=pd.read_csv('credit_bank.csv')


X=pd.read_csv('credit_bank.csv').values

from sklearn.cluster import KMeans
inertia_values =[]

clusters = [1,2,3,4,5,6,7,8,9,10]

for i in clusters :
    kmeans=KMeans(algorithm='auto', copy_x=True,init='k-means++',max_iter=300, n_clusters=i,n_init=10, n_jobs=1,
              precompute_distances='auto', random_state=None, tol=1e-4, verbose=0)
    
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
   
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.

print(kmeans.cluster_centers_)



import seaborn as sns

sns.pointplot(x = clusters, y = inertia_values )
plt.xlabel('Nombre de Clusters')
plt.ylabel("valeur d'Inertie")
plt.title(" Nombre de Clusters Vs. valeur d' Inertie")
plt.show()



kmeans=KMeans(n_clusters=3)

y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans== 0, 0], X[y_kmeans== 0, 1], s=100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans== 1, 0], X[y_kmeans== 1, 1], s=100, c='red',    label='Cluster 2')
plt.scatter(X[y_kmeans== 2, 0],X[y_kmeans==  2, 1], s=100, c='magenta', label= 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters de demandeur_credit')
plt.xlabel('epargne en millier')
plt.ylabel('score_bank')
plt.legend()
plt.show()





