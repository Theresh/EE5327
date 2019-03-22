import numpy as np
import matplotlib.pyplot as plt



x1 = np.linspace(-1,3,20)
x2 = (9-9*x1)/13
plt.plot(x1,x2,label='$f(x)=9x_1 + 13x_2 = 9$')

x1 = np.linspace(-1,3,2000)
x2 = 1 - x1

plt.plot(x1,x2,label='$g1(x)=x_1 + x_2 >= 1$')

x1 = np.linspace(0,3,2000)
x2 = 1 - x1
plt.fill_between(x1,x2,4, x2>0 ,color = 'grey')
plt.fill((1,3,3,1,1),(0,0,4,4,0),color = 'grey')


plt.plot(1.0,0,'o')
plt.annotate('%.2f)' %0.0, xy=(1.0,0), xytext=(30,0), textcoords='offset points')
plt.annotate('(%.2f,' %1.0, xy=(1.0,0))
plt.plot((0,3),(0,0),label='$g2(x)=x_1>= 0$')
plt.plot((0,0),(0,4),label='$g3(x)=x_2 >= 0$')

plt.grid()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = '1')
