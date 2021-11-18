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

def unisam1():
    return random.choice([0,1,2,3,4,5,6,7,8,9])      


def unisam(l):
    return random.sample([0,1,2,3,4,5,6,7,8,9],l)

def svrgunisam(p):
    tmp=random.uniform(0,1)
    if tmp<p:
        return 1
    else:
        return 0

def expect0(xx,yy,zz,dd,jj,pp):  
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
            index  = unisam1()
            gradi = funct[index].grad(x)
            x = x - stepsizes*0.1*gradi
            traj.append(x)
        dis = dis + fullgrad(x)[0,0]**2+fullgrad(x)[1,0]**2
    return c*dis/num_round


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
            index  = unisam1()
            gradi = funct[index].grad(x)-funct[index].grad(y)+fullgrad(y)
            x = x - stepsizes*0.1*gradi
            traj.append(x)
            if svrgunisam(prob):
                y = x
            else:
                y = y
        dis = dis + fullgrad(x)[0,0]**2+fullgrad(x)[1,0]**2
    return c*dis/num_round

def expect2(xx,yy,zz,dd,jj,pp,ll):  
    tau = xx 
    iter = yy
    num_round = zz
    initial = dd
    stepsizes = jj
    dis = 0
    prob = pp
    l=ll
    for rd in range(num_round):
        x = initial
        y = fullgrad(x)
        traj = []
        traj.append(x)
        for i in range(iter):
            index  = unisam(ll)
            if i>5:
                for k in index:
                    gradi = (ll**-1)*(funct[k].grad(x)-funct[k].grad(traj[-2]))
                if svrgunisam(prob):
                    y=fullgrad(x)
                    x = x - stepsizes*0.1*y
                else:
                    y=y+gradi
                    x = x - stepsizes*0.1*y
            else:
                x = x -stepsizes*0.1*fullgrad(x)
            traj.append(x)   
        dis = dis + fullgrad(x)[0,0]**2+fullgrad(x)[1,0]**2
    return c*dis/num_round



dis_final0=[]
dis_final1=[]
dis_final2=[]

stepsize = 100**(-1)

init = matrix([[-0.7],[-0.2]])
x = 0
num_r = 10
dis = []
tauu = 10

iter = [300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]

for i in iter:
    print(i)
    dis_final0.append(expect0(10,i,num_r,init,stepsize,0))
    dis_final1.append(expect1(10,i,num_r,init,stepsize,0.01))
    dis_final2.append(expect2(10,i,num_r,init,stepsize,0.5,3))
    

plt.plot(iter,dis_final0,linewidth = 1,marker= '*',markersize = 10, label = 'L-SVRG-Uniform p=0.9')
plt.plot(iter,dis_final1,linewidth = 1,marker= '+',markersize = 10, label = 'L-SVRG-Uniform, p=0.01')
plt.plot(iter,dis_final2,linewidth = 1,marker= 'o',markersize = 10, label = 'L-SVRG-Uniform, p=0.5,tau=3')
 
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error ')
plt.title('Stepsize = 0.05')
plt.show()