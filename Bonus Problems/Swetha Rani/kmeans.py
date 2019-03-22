import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

X=np.genfromtxt('Xsvm1.csv')
S=X.shape
#X1=[X,X]
c1,c2,c3,c4=100,90,38,5

for k in range(40):
  
    clu1=[]
    clu2=[]
    clu3=[]
    clu4=[]
    for i in range(48):
        d1=distance.euclidean(X[i],c1)
        d2=distance.euclidean(X[i],c2)
        d3=distance.euclidean(X[i],c3)
        d4=distance.euclidean(X[i],c4)
        if d1<=d2 and d1<=d3 and d1<=d4:
            clu1.append(X[i])
        elif d2<=d1 and d2<=d3 and d2<=d4:
            clu2.append(X[i])
        elif d3<=d1 and d3<=d2 and d3<=d4:
            clu3.append(X[i])
        else: 
            clu4.append(X[i])
    
    c1=np.mean(clu1)
    c2=np.mean(clu2)
    c3=np.mean(clu3)
    c4=np.mean(clu4)

plt.scatter(clu1,clu1)
plt.scatter(clu2,clu2)
plt.scatter(clu3,clu3)
plt.scatter(clu4,clu4)
plt.show()
plt.savefig('out put.jpg')
print('A grade')
print(clu1)
print('B grade')
print(clu2)
print('c grade')
print(clu3)
print('D grade')
print(clu4)
