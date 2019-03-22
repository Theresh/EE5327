import cvxpy as cp
import numpy as np

x = cp.Variable((4,4),nonneg=True)

constraints = [x[0,0]+x[0,1]+x[0,2]+x[0,3]==1,
               x[1,0]+x[1,1]+x[1,2]+x[1,3]==1,
               x[2,0]+x[2,1]+x[2,2]+x[2,3]==1,
               x[3,0]+x[3,1]+x[3,2]+x[3,3]==1,
               x[0,0]+x[1,0]+x[2,0]+x[3,0]==1,
               x[0,1]+x[1,1]+x[2,1]+x[3,1]==1,
               x[0,2]+x[1,2]+x[2,2]+x[3,2]==1,
               x[0,3]+x[1,3]+x[2,3]+x[3,3]==1]

obj = cp.Minimize(
    5*x[0,0]+3*x[0,1]+2*x[0,2]+5*x[0,3]+
    7*x[1,0]+9*x[1,1]+2*x[1,2]+3*x[1,3]+
    4*x[2,0]+2*x[2,1]+3*x[2,2]+2*x[2,3]+
    5*x[3,0]+7*x[3,1]+7*x[3,2]+5*x[3,3])

prob = cp.Problem(obj,constraints)
prob.solve()

print np.round(prob.value)

for i in x:
    for j in i:
        print np.round(j.value),
    print
