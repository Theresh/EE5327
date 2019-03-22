import cvxpy as cp

x11 = cp.Variable()
x12 = cp.Variable()
x13 = cp.Variable()
x14 = cp.Variable()
x21 = cp.Variable()
x22 = cp.Variable()
x23 = cp.Variable()
x24 = cp.Variable()
x31 = cp.Variable()
x32 = cp.Variable()
x33 = cp.Variable()
x34 = cp.Variable()
constraints = [x11+x12+x13+x14<=22,
               x21+x22+x23+x24<=15,
               x31+x32+x33+x34<=8,
               x11+x21+x31>=7,
               x12+x22+x32>=12,
               x13+x23+x33>=17,
               x14+x24+x34>=9,
               x11>=0,x12>=0,x13>=0,x14>=0,
               x21>=0,x22>=0,x23>=0,x24>=0,
               x31>=0,x32>=0,x33>=0,x34>=0]
obj = cp.Minimize(6*x11+3*x12+5*x13+4*x14+
                  5*x21+9*x22+2*x23+7*x24+
                  5*x31+7*x32+8*x33+6*x34)

prob = cp.Problem(obj,constraints)
prob.solve()

print prob.value
print "%0.2f"%x11.value,"%0.2f"%x12.value,"%0.2f"%x13.value,"%0.2f"%x14.value
print "%0.2f"%x21.value,"%0.2f"%x22.value,"%0.2f"%x23.value,"%0.2f"%x24.value
print "%0.2f"%x31.value,"%0.2f"%x32.value,"%0.2f"%x33.value,"%0.2f"%x34.value
