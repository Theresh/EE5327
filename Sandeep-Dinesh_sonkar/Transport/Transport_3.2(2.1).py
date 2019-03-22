import cvxpy as cp

x11 = cp.Variable()
x12 = cp.Variable()
x13 = cp.Variable()
x21 = cp.Variable()
x22 = cp.Variable()
x23 = cp.Variable()
x31 = cp.Variable()
x32 = cp.Variable()
x33 = cp.Variable()

constraints = [x11+x12+x13<=40,
               x21+x22+x23<=60,
               x31+x32+x33<=10,
               x11+x21+x31>=40,
               x12+x22+x32>=50,
               x13+x23+x33>=20,
               x11>=0,x12>=0,x13>=0,
               x21>=0,x22>=0,x23>=0,
               x31>=0,x32>=0,x33>=0]
obj = cp.Minimize(2*x11+1*x12+2*x13+
                  9*x21+4*x22+7*x23+
                  1*x31+2*x32+9*x33)

prob = cp.Problem(obj,constraints)
prob.solve()

print prob.value
print "%0.1f"%x11.value,"%0.1f"%x12.value,"%0.1f"%x13.value
print "%0.1f"%x21.value,"%0.1f"%x22.value,"%0.1f"%x23.value
print "%0.1f"%x31.value,"%0.1f"%x32.value,"%0.1f"%x33.value
