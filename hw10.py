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
from random import sample

#function
a = [matrix([[0.1]]), matrix([[0.424466]]), matrix([[0.77981303]]), matrix([[0.20033184]]), matrix([[0.51116473]]), matrix([[0.2604399]]), matrix([[0.97100656]]), matrix([[0.21263449]]), matrix([[0.26417151]]), matrix([[0.15995097]])]

b = [matrix([[0.4231786]]), matrix([[0.524466]]), matrix([[0.17981303]]), matrix([[0.50033184]]), matrix([[0.71116473]]), matrix([[0.0604399]]), matrix([[0.37100656]]), matrix([[0.91263449]]), matrix([[0.66417151]]), matrix([[0.65995097]])]

class func:
    def __init__(self, a,b):
        self.a = a
        self.b = b
        
    def value(self,x):
        return  math.sin(x[0,0]+self.a)+math.cos(x[1,0]+self.b)
                
    def grad(self,x):
        return (matrix([[math.cos(x[0,0]+self.a)],[-math.sin(x[1,0]+self.b)]]))


funct = []
for i in range(10):
    funct.append(func(a[i],b[i]))   

def fullgrad(x):
    grad = matrix([[0],[0]])
    for i in range(10):
        grad = grad + funct[i].grad(x)
    return 0.1*grad
        


def unisam():
    return random.choice([0,1,2,3,4,5,6,7,8,9])

def svrgunisam(p):
    tmp=random.uniform(0,1)
    if tmp<p:
        return 1
    else:
        return 0

def expect1(xx,yy,zz,dd,jj,pp):  
    tau = xx 
    iter = yy
    num_round = zz
    initial = dd
    stepsizes = jj
    dis = 0
    prob = pp
    for rd in range(num_round):
        x = initial
        y = x
        traj = []
        traj.append(x)
        for i in range(iter):
            index  = unisam()
            gradi = funct[index].grad(x)-funct[index].grad(y)+fullgrad(y)
            x = x - stepsizes*0.1*gradi
            traj.append(x)
            if svrgunisam(prob):
                y = x
            else:
                y = y
    return traj




stepsize = 40**(-1)
init = matrix([[-0.5],[-0.2]])
x = 0
num_r = 1
dis = []
tauu = 10


dis_final=expect1(10,3000,1,init,stepsize,0)
dis_final1=expect1(10,3000,1,init,stepsize,0.9)
x =[]
y =[]
xx=[]
yy=[]
for k in range(3000):
    x.append(dis_final[k][0,0])
    y.append(dis_final[k][1,0])
    xx.append(dis_final1[k][0,0])
    yy.append(dis_final1[k][1,0])


print(x)
print(xx)
plt.plot(x,y,linewidth = 0.5, label = 'L-SVRG-Uniform p=0.01')
plt.plot(xx,yy,linewidth = 0.5, label = 'L-SVRG-Uniform, p=0.9')
 
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory')
plt.show()
