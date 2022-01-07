# -*- coding: utf-8 -*-
"""
Created on Mon Sep  20 19:29:31 2021

@author: Lukang Sun
"""

from matplotlib.legend import Legend
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy import matrix
from numpy import *
import math
import random
c=1
#function
a = [9.9,7.2,3.6,8.9,5.3,-9.9,-7.2,-3.6,-8.9,-5.3]

b = [10,2,4,6,4,10,2,4,6,4]

class func:
    def __init__(self, a,b):
        self.a = a
        self.b = b
        self.c=2.1
    def value(self,x):
        return  10*x**2+self.a*abs(x-self.b)**self.c
                
    def grad(self,x):
        if x-self.b > 0:
            return 20*x+self.c*self.a*(x-self.b)**(self.c-1)

        else:
            return 20*x-self.c*self.a*abs(x-self.b)**(self.c-1)



funct = []
for i in range(10):
    funct.append(func(a[i],b[i]))   

def fullgrad(x):
    
    return 20*x

def unisampling():
    return random.choice([0,1,2,3,4,5,6,7,8,9])      


def Probability(p):
    tmp=random.uniform(0,1)
    if tmp<p:
        return 1
    else:
        return 0

def Page(init,iter,stepsizes,prob):  
    
    y = init
    traj_dis = []
    g = fullgrad(init)
    x = init - stepsizes*g
    
    for i in range(iter):
        if Probability(prob):
            y = x
            x = x - stepsizes*fullgrad(x)
        else:

            tmp = unisampling()
            g = g + funct[tmp].grad(x)-funct[tmp].grad(y)

            y = x
            x = x - stepsizes*g
        traj_dis.append(x**2)
    return traj_dis



def Gradientdescent(init,iter,stepsizes):  
    x = init
    traj_dis = []
    for i in range(iter):
        
        x = x - stepsizes*fullgrad(x)
        traj_dis.append(x**2)
            
    return traj_dis

init = 5
stepsize = 0.03
iter = 1000


dis_final0=Gradientdescent(init,iter,stepsize)
dis_final1=Page(init,iter,stepsize,0.05)
    

plt.plot(range(iter),dis_final0,linewidth = 1,marker= '*',markersize = 10, label = 'Hessian')
plt.plot(range(iter),dis_final1,linewidth = 1,marker= '.',markersize = 10, label = 'PAGE')

 
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error ')
plt.title('Stepsize = 0.03')
plt.show()