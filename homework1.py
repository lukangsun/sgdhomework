# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:29:31 2021

@author: Lukang Sun
"""


import numpy as np
import matplotlib.pyplot as plt
a=0.01
b=10000
c=b/a
def f(x,y):
    
    return a*x**2+b*y**2

def gradient(x,y):
    
    return [2*a*x,2*b*y]

#parameter

iter = 800
initial = [5,0.000005]
stepsize = 1/b
trajx = []
trajy = []

#gradient descent iteration

x = initial
trajx.append(x[0])
trajy.append(x[1])

for i in range(iter):
    
    x[0] = x[0] - stepsize*gradient(x[0],x[1])[0]
    x[1] = x[1] - stepsize*gradient(x[0],x[1])[1]
    trajx.append(x[0])
    trajy.append(x[1])
    
#plot
import math
print(((c+3)/4)*math.log(25*(trajx[iter])**(-2)))
x=np.linspace(-1,6,1000)
y=np.linspace(-1,6,1000)
X,Y = np.meshgrid(x,y)
z=f(X,Y)
plt.contour(x,y,z)
plt.plot(trajx,trajy,color='r',marker= '.',linewidth = 0.3)
#plt.scatter(trajx,trajy,color='b')
#plt.grid(True)
#plt.tight_layout()
plt.show()
