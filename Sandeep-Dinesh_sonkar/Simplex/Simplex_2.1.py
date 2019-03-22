import cvxpy as cp

a = cp.Variable()
b = cp.Variable()

constraints = [a+b<=10,2*a+3*b<=25,a+5*b<=35]
obj = cp.Maximize(6*a+8*b)

prob = cp.Problem(obj,constraints)
prob.solve()

print prob.value
print a.value
print b.value

