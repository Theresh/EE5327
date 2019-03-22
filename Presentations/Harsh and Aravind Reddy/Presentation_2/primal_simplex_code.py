from cvxpy import *
from numpy import matrix
import numpy as np

x = Variable((3,1))
A = matrix([
	[-1.0, -2.0 , 0.0],
	[0.0, 1.0 , 7.0], 
	[1.0, -3.0 , 5.0],
	[-1.0, 3.0 , -5.0], 
	[0.0, -1.0 , 0.0],	
	])
b = matrix([
	[-4.0],
	[5.0],
	[6.0],
	[-6.0],
	[0.0]
	])

f = x[0,0]+x[1,0]+2*x[2,0]
obj = Minimize(f)

constraints = [A*x <= b ]

#solution
Problem(obj, constraints).solve()
print("Min Cost = ", f.value)
print("Minima is at x = \n",x.value)