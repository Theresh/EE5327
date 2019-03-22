import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# dataframe x- total marks
df = pd.DataFrame({ 
    'x' : [17.2,0,0,76.7,66.7,74.2,10,10,95,26.1,15,99.2,99.2,0,0,0,0,96.4,10,27.7,88.4,86.7,84.2,26.9,91.7,95.7,94.2,96.7,17.2,0,93.2,74.7,0,8,90.1,92.5,59.2,102,90,92.5,97.5,50.4,25.2,96,86.5,72.6,60.4,63.4],
    'y' : [0]*48 
})
k = 8 #no of grades - FR,D,C-,C,B-,B,A-,A
colmap = {1:'r',2:'g',3:'b',4:'y',5:'k',6:'c',7:'m',8:'#b22222'}
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k)

kmeans.fit(df)
labels = kmeans.predict(df)

centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5,5))


colors =map(lambda x: colmap[x+1],labels)
colors1 = list(colors)

plt.scatter(df['x'],df['y'],color = colors1,edgecolor='k')

for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color = colmap[idx+1])

plt.xlim(0,110)

plt.show()
#FIG divides the data into 8 clusters (grades) ,each cluster is indicated by a different color
# Centroids of each cluster are demarcated from its members by coloring without a border
# Grades can be allotted for each cluster starting with the largest grade to the right most cluster