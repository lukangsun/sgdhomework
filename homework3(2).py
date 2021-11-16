# -*- coding: utf-8 -*-
"""
Created on Mon Sep  20 19:29:31 2021

@author: Lukang Sun
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

#define function， gradient and random index

L = [2,4,6,3,2,9,6,7,11,45]               #length = 10
L_sum = 95                                #summation of L
i=0
tmp =[]
for j in L:
    tmp.extend([i]*j)
    i = i+1
Long_list = tmp                          # used for importance sampling

class func:
    def __init__(self, smoothness):
        self.smoothness = smoothness
        
    def value(self,x,y):
        return  0.5*self.smoothness*(0.2*x**2+y**2)

    def grad(self, x, y):
        return [0.2*self.smoothness*x,self.smoothness*y]


funct = []
for i in L:
    funct.append(func(i))               #define each function i and its gradient

def fun_sum(x,y):
    return 0.1*0.5*L_sum*(0.2*x**2+y**2)    #return total function

def us():                               #uniform sampling
    return random.randint(0,9)
def importsamp():                       #importance sampling
    return random.choice(Long_list)

#hyper parameter
iter = 10
epsilon = 10**(-5)
stepsize_us = 45**(-1)
stepsize_is = 9.5**(-1) 
initial = [4,5]
x = [0,0]
y = [0,0]
num_round = 100
dis_us = 0
dis_is = 0
#iteration

for rd in range(num_round):
    x[0] = initial[0]
    x[1] = initial[1]
    y[0] = initial[0]
    y[1] = initial[1]
    trajx_us = []
    trajx_us.append(x[0])
    trajy_us = []
    trajy_us.append(x[1])
    trajx_is = []
    trajx_is.append(x[0])
    trajy_is = []
    trajy_is.append(x[1])
    
    for i in range(iter):
    
        index_us = us()

        x[0] = x[0] - stepsize_us*funct[index_us].grad(x[0],x[1])[0]
        x[1] = x[1] - stepsize_us*funct[index_us].grad(x[0],x[1])[1]
        trajx_us.append(x[0])
        trajy_us.append(x[1])

        index_is = importsamp()

        y[0] = y[0] - stepsize_is*(9.5/(L[index_is]))*funct[index_is].grad(y[0],y[1])[0]
        y[1] = y[1] - stepsize_is*(9.5/(L[index_is]))*funct[index_is].grad(y[0],y[1])[1]
        trajx_is.append(y[0])
        trajy_is.append(y[1])
    dis_us = dis_us + trajx_us[iter]**2+trajy_us[iter]**2
    dis_is = dis_is + trajx_is[iter]**2+trajy_is[iter]**2


print(dis_us/num_round,dis_is/num_round)


#plot

x=np.linspace(-1,6,1000)
y=np.linspace(-1,6,1000)
X,Y = np.meshgrid(x,y)
z=fun_sum(X,Y)
#plt.contour(x,y,z,10)
plt.contour(x,y,z,50,linewidths = 0.2)
#plt.contourf(h, levels=[10, 30, 50],colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
plt.plot(trajx_us,trajy_us,color='r',marker= '.',markersize = 3,linewidth = 0.6,label='Uniform')
plt.plot(trajx_is,trajy_is,color='b',marker= '*',markersize = 3,linewidth = 0.6,label='Importance')
#plt.plot(trajx_is,trajy_is,color='g',marker= '.',markersize = 0.2,linewidth = 0.6)
plt.title("Iteration = %d, Uniform Error = %f, Importance Error = %f" %(iter,dis_us/num_round, dis_is/num_round))  #figure 2
#plt.title("sigma = %f, E = %f, T = %f" %(sigma, dis/num_round, stepsize*sigma**2/mu))  #figure 1
#plt.scatter(trajx_is,trajy_is,color='b')
#plt.grid(True)
#plt.tight_layout()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()