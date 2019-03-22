import numpy as np 
from sklearn.cluster import KMeans  


def Predict(x,centers):
	grads = []
	centers = centers.reshape(-1)
	for elem in x:
		if(elem>=centers[-1]):
			grads.append("A")
		if(elem>=centers[-2] and elem<centers[-1]):
			grads.append("A-")
		if(elem>=centers[-3] and elem<centers[-2]):
			grads.append("B")
		if(elem<centers[-3]):
			grads.append("B-")
	return grads

inp_data = [17.2,0,0,76.7,66.7,74.2,10,10,95,26.1,15,99.2,99.2,0,0,0,0,96.4,10,27.7,88.4,86.7,84.2,26.9,91.7,95.7,94.2,96.7,17.2,0,93.2,74.7,0,8,90.1,92.5,59.2,102,70,92.5,97.5,50.4
,25.2
,96
,86.5
,72.6
,60.4
,63.4]

X = np.array(inp_data)
X = X.reshape(-1,1)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
centers = np.sort(centers.reshape(-1))
centers = centers.reshape(-1,1)
predictions = Predict(inp_data,centers)
print(predictions)