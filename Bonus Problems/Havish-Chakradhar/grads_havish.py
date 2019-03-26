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

inp_data = []

X = np.array([76.7,
76.7,
74.2,
95,
68.4,
73.4,
72.6,
99.2,
99.2,
98.4,
88.4,
86.7,
84.2,
91.7,
95.7,
94.2,
96.7,
93.2,
76.7,
92.1,
92.5,
61.2,
102,
92,
92.5,
97.5,
66.8,
25.2,
96,
86.5])#np.array(inp_data)
X = X.reshape(-1,1)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
centers = np.sort(centers.reshape(-1))
centers = centers.reshape(-1,1)
predictions = Predict(inp_data,centers)
print(predictions)