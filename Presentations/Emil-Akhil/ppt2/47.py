import cvxpy as cp
import numpy as np

x = cp.Variable(2)

constraints = [
				2*x[0] + x[1] >= 8,
				2*x[0] + 5*x[1] >= 10,				
				x >=0
			  ]

objective = cp.Minimize(x[0] + x[1])

problem = cp.Problem(objective,constraints)

problem.solve()

print((problem.value))
X = (x.value)
print(X)