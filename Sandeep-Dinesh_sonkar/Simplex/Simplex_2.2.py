import cvxpy as cp

a = cp.Variable()
b = cp.Variable()

constraints = [a+b<=5,3*a+2*b<=12]
obj = cp.Maximize(6*a+5*b)

prob = cp.Problem(obj,constraints)
prob.solve()

print "%0.2f"%prob.value
print "%0.2f"%a.value
print b.value

