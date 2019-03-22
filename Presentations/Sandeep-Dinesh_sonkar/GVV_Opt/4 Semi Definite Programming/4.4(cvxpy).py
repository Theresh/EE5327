import cvxpy as cp

X = cp.Variable((2,2),PSD = True)
constraints = [2*X[0,0]+3*X[0,1]+X[1,1]==7,X[0,0]+X[0,1]>=1,X[0,0]>=0,X[0,1]>=0,X[1,1]>=0]
obj = cp.Minimize(-X[0,0]-2*X[0,1]-5*X[1,1])

prob = cp.Problem(obj,constraints)
prob.solve()

print "%0.2f"%prob.value
print "x11 = %0.2f"%X.value[0][0]
print "x12 = %0.2f"%X.value[0][1]
print "x22 = %0.2f"%X.value[1][1]

