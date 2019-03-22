from cvxopt import matrix
from cvxopt import solvers

c = matrix([-1.,-2.,-5.])
G = [ matrix([[-1., 0., 0., 0.],
			  [ 0., -1., -1., 0.],
			  [0.,  0.,  0., -1.]])  ]
G += [ matrix([[-1., 0., 0., 0.],
			   [ -1., -1., 0., -1.],
			   [0.,  0.,  -1., 0.]]) ]
Aval = matrix([2.,3.,1.],(1,3))
bval = matrix([7.])
h = [ matrix([[0., 0.], [0., 0.]]) ]
h += [ matrix([[-1., 0.], [0., 0.]]) ]
sol = solvers.sdp(c, Gs=G, hs=h,A=Aval, b=bval)

print(sol['x'])        

print("constrain 1 : 2x_11 + 3x_12 + x_22 = ",2*sol['x'][0] + 3*sol['x'][1] + sol['x'][2])
print("constrain 2 : x_11 + x_12 = ",sol['x'][0] + sol['x'][1])
print("Minimum value = ",-1*sol['x'][0] + -2*sol['x'][1] + -5*sol['x'][2])