from scipy.optimize import linprog

C = [-40,-100]      #cost function
A = [[10,5],[4,10],[2,3]]   #Constraint matrix
B = [2500,2000,900]     #RHS of constraints
x0_bounds = (0,None)    #making sure x1 and x2 are >=0
x1_bounds = (0,None)
#call the lin prog function from the library we imported
res = linprog(C,A_ub=A,b_ub=B,bounds=(x0_bounds,x1_bounds),options={"disp":True})
