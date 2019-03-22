import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def fnc(X):
    return (X[0]* X[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection=Axes3D.name)
x = y = np.linspace(-50,50,1000)
X, Y = np.meshgrid(x, y)
Z = fnc([X,Y])
ax.plot_surface(X, Y, Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=15, azim=-118)
plt.show()
