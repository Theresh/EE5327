
import cvxpy as cvx
from numpy import matrix, round, eye
import numpy
#Create Variable
X = cvx.Variable()
Y = cvx.Variable()

constraints = [-X+2*Y<=2,
                3*Y-X>=0,
                2*X+3*Y<=10,
                X>=0,
                Y>=0]

obj = cvx.Maximize(3*X+9*Y)

prob = cvx.Problem(obj,constraints)
prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", X.value, Y.value)
