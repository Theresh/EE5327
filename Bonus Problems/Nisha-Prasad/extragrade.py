#Extra Grade Question
#EE18MTECH11001 Prasad Gaikwad
#EE18MTECH11002 Nisha Akole

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

fullpath = 'Opt_FinalMarks.csv'
df = pd.read_csv(fullpath,header=None)
data = list(df[1])
norm_data = (data - np.mean(data))/np.var(data)

model = KMeans(n_clusters= 4)
model.fit(norm_data.reshape(-1,1))
predictions = model.predict(norm_data.reshape(-1,1))

dictt = {'0':'A','1':'B','2':'C','3':'D','4':'E'}
name =[]
for i in list(predictions):
    for k, j in dictt.iteritems():
        if int(k) == i:
            name.append(j)
            
for i in range(len(data)):
     print('Roll No:', df[0][i],'Grade: Class',name[i])

data1 = (np.matrix(data)).reshape(-1,1)
distortions = []
K = range(1,7)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data1)
    kmeanModel.fit(data1)
    distortions.append(sum(np.min(cdist(data1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data1.shape[0])

# Plot the elbow to choose optimum
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()