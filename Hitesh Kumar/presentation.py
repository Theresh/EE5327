import numpy as np
import cvxpy as cp
import operator
import matplotlib.pyplot as plt
import random
j=[]
x=[]
delta=[]
fin=[]
m=100
n=4
for k in range(60,100,1):
	a=[]
	b=[]
	su=np.zeros((n,n))
	c=np.zeros((n,n))
	z=cp.Variable((m,1))			#take m z[i]'s				
	for i in range (0,m,1):
		a.append(np.random.rand(n,1))		#vectors a[i] taken from gaussian distribution
		b.append(np.matmul(a[i],a[i].T))	#computing matrix for each info (error covariance matrix)
		c=c+b[i]*z[i]
	constraints=([z[0]<=1,z[0]>=0])
	for i in range (1,m,1):
		constraints=constraints+[z[i]<=1,z[i]>=0] #constraints are linear
	obj=cp.Maximize(cp.log_det(c))		#objective function
	prob = cp.Problem(obj, constraints)	#get the sub-optimal solution
	prob.solve() 
	for i in range(0,m,1):
		j.append([z[i].value,a[i]])		#vector pair of z[i] with corresponding a[i]
	j.sort(reverse=True)				#set z[i] in decreasing order in list j
	for i in range(0,k,1):
		su=su+ (np.matmul(j[i][1],j[i][1].T))	#matrix for greastest k values
	ans=np.log(np.linalg.det(su))
	fin.append(ans)
	delta.append(abs(prob.value-ans))	
	x.append(k)
plt.plot(x,fin)
plt.show()
plt.plot(x,delta)
plt.show()