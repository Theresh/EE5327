import numpy as np
import matplotlib.pyplot as plt
c=3
x = np.linspace(-4, 4, 256 , endpoint = True )
y = x**3 - 3*x*c

plt.plot(x, y)

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()