import cvxpy as cp

x = cp.Variable(2)

constraints = [x[0]+x[1]<=2,5*x[0]+2*x[1]<=10,3*x[0]+8*x[1]<=12,x>=0]
obj = cp.Maximize(5*x[0]+3*x[1])

prob = cp.Problem(obj,constraints)
prob.solve()

print '%0.2f'%prob.value
print '%0.2f'%x[0].value
print '%0.2f'%x[1].value
