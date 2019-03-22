import cvxpy as cp

x1 = cp.Variable()
x2 = cp.Variable()

constraints = [x1+x2<=5,3*x1+2*x2<=12,x1>=0,x2>=0]
obj = cp.Maximize(6*x1+5*x2)

prob = cp.Problem(obj,constraints)
prob.solve()

print prob.value
print x1.value
print x2.value
