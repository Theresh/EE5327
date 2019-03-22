from cvxpy import *
from numpy import matrix
import numpy as np

x = Variable((3,1))
A = matrix([
	[1.0, 0.0 , 1.0],
	[2.0, 1.0 , -3.0], 
	[0.0, 7.0 , 5.0],
	[0.0, -7.0 , -5.0], 
	[-1.0, 0.0 , 0.0],	
	[0.0, 1.0 , 0.0],
	])
b = matrix([
	[1.0],
	[1.0],
	[2.0],
	[-2.0],
	[0.0],
	[0.0]
	])

f = 4*x[0,0]+5*x[1,0]+6*x[2,0]
obj = Maximize(f)

constraints = [A*x <= b ]

#solution
Problem(obj, constraints).solve()
print("Max Cost = ", f.value)
print("Max is at x = \n",x.value)