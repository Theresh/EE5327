import cvxpy as cp

x = cp.Variable(2)

constraints=[cp.norm(x,2)<=5**0.5,x>=0]
obj = cp.Maximize(cp.geo_mean(x))

prob = cp.Problem(obj, constraints)
prob.solve()
print prob.value
print x.value
