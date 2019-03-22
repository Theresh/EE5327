#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:49:20 2019

@author: ravilal
"""

import numpy as np
import matplotlib.pyplot as plt


#Plotting x^2
x = np.linspace(-5,6,50)#points on the x axis
f=x*x#Objective function
plt.plot(x,f,color=(1,0,1))
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$x^2$')


#Plot commands
#Plotting line segments with x>-5 and y=x^2 
#color used to color each line with a different color
plt.plot([1,4],[1*1,4*4],color=(1,0,0),marker='o',label="($(1)$,$(1)^2$)-($(4)$,$(4)^2$)")
plt.plot([2,6],[2*2,6*6],color=(0,1,0),marker='o',label="($(2)$,$(2)^2$)-($(6)$,$(6)^2$)")
plt.plot([3,5],[3*3,5*5],color=(0,0,1),marker='o',label="($(3)$,$(3)^2$)-($(5)$,$(5)^2$)")
plt.legend(loc=2)
#plt.savefig('../figs/1.3.eps')
plt.show()#Reveals the plot