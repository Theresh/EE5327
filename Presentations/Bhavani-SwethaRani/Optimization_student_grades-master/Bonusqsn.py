import csv
import numpy as np
from sklearn.cluster import KMeans

#k=7,gardesA,A-,B,B-,C,C-,D
k=7
filename = "New_Mark_Sheet.csv"
X = []
       
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        X.append(float(row[1]))
        
#print(X)

X1= np.array(list(X)).reshape(1,-1)
#No of iterations by default 300
#Distances is 2-norm distances
#tol=Relative tolerance with regards to inertia to declare convergencedefault=1e-4
kmeans = KMeans(n_clusters=7, random_state=0).fit(X1.T)
centroids=kmeans.cluster_centers_ 
#Grades output taken with  centroid indeces
#{A,A-,B,B-,C,C-,D}={0,4,2,5,3,1,6}
#print(centroids)
G=kmeans.predict(X1.T)
#Grades allotted for each student
#print(Grades)
for i in range(len(G)):
	if G[i]==0:
		Grades="A"
	elif G[i]==1:
		Grades='C-'
	elif G[i]==2:
		Grades='B'
	elif G[i]==3:
		Grades='C'
	elif G[i]==4:
		Grades='A-'
	elif G[i]==5:
		Grades='B-'
	else:
		Grades='D'
	print(Grades)
#The output in the last column of csv file
