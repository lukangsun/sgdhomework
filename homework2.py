# -*- coding: utf-8 -*-
"""
Created on Mon Sep  20 19:29:31 2021

@author: Lukang Sun
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

#define function and gradient

mu=0.02
L=2
condition_num=mu/L
sigma = 0.1

def f(x,y):
    
    return 0.5*(mu*x**2+L*y**2)

def gradient(x,y):
    
    rand=random.normal(loc=0,scale=sigma,size=(1,2))
    return [mu*x+rand[0][0],L*y+rand[0][1]]

#parameter
epsilon = 0.00001
iter = 5*math.ceil((-(1+3*condition_num)/(4*condition_num))*math.log(epsilon)) #iter = 5 times the minmum iteration step as derived from question3
initial = [5,5]
x = [0,0]
stepsize = ((2*1+2*condition_num)/(L*(1+3*condition_num)))/6
num_round = 100
dis = 0


#gradient descent iteration
for rd in range(num_round):
    x[0] = initial[0]
    x[1] = initial[1]
    trajx = []
    trajy = []
    trajx.append(x[0])
    trajy.append(x[1])

    for i in range(iter):
        
        x[0] = x[0] - stepsize*gradient(x[0],x[1])[0]
        x[1] = x[1] - stepsize*gradient(x[0],x[1])[1]
        trajx.append(x[0])
        trajy.append(x[1])
    dis = dis + trajx[iter]**2+trajy[iter]**2

print(dis/num_round,stepsize*sigma**2/mu,stepsize)

        
#plot

x=np.linspace(-3,6,1000)
y=np.linspace(-3,6,1000)
X,Y = np.meshgrid(x,y)
z=f(X,Y)
#plt.contour(x,y,z,10)
plt.contour(x,y,z,10,linewidths = 0.2)
#plt.contourf(h, levels=[10, 30, 50],colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
plt.plot(trajx,trajy,color='r',marker= '.',markersize = 0.2,linewidth = 0.6)
plt.title("stepsize = %f, E = %f, T = %f" %(stepsize, dis/num_round, stepsize*sigma**2/mu))  #figure 2
#plt.title("sigma = %f, E = %f, T = %f" %(sigma, dis/num_round, stepsize*sigma**2/mu))  #figure 1
#plt.scatter(trajx,trajy,color='b')
#plt.grid(True)
#plt.tight_layout()
plt.show()







