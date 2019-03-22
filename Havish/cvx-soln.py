from cvxpy import *
from numpy import matrix

A = matrix([ [1.0, 1.0], [3.0, 2.0 ]])
b = matrix([ 5.0, 12.0 ])
c = matrix([ 6.0, 5.0 ])

x = Variable((2,1),nonneg=True)
#Cost function
f = c*x
obj = Maximize(f)
#Constraints
constraints = [A*x <= b.transpose()]

#solution
Problem(obj, constraints).solve()

(x1,x2) = x.value

print("Maximum value is " + str(f.value[0][0]))
print("x1: " + str(x1[0]))
print("x2: " + str(x2[0]))

