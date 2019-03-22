import cvxpy as cp

w = cp.Variable(2)
d = cp.Variable()

constraints = [2*w[0]+w[1]+d>=1,0.8*w[0]-0.6*w[1]+d<=-1]
obj = 0.5*cp.Minimize(cp.square(w[0])+cp.square(w[1]))

prob = cp.Problem(obj,constraints)
prob.solve()

print prob.value
print w.value
