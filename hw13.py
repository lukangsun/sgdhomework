# -*- coding: utf-8 -*-
"""
Created on Mon Sep  20 19:29:31 2021

@author: Lukang Sun
"""
import matplotlib
from matplotlib.legend import Legend
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy import matrix
from numpy import *
import math
import heapq
import random
from numpy import linalg as la
innitialpoint = matrix([[1],[1]])
a=50
b=300
c=200
d=7
#function
def f_1(x):
    return a*x[0,0]**2+b*x[1,0]**2+math.sin(x[0,0])

def f_2(x):
    return c*x[0,0]**2+d*x[1,0]**2+math.cos(x[1,0])

def grad_1(x):
    return (matrix([[2*a*x[0,0]+math.cos(x[0,0])],[2*b*x[1,0]]]))

def grad_2(x):
    return (matrix([[2*c*x[0,0]],[2*d*x[1,0]-math.sin(x[1,0])]]))

def hess_1(x):
    return matrix([[2*a-math.sin(x[0,0]),0],[0,2*b]])

def hess_2(x):
    return matrix([[2*c,0],[0,2*d-math.cos(x[1,0])]])

def unisam1():
    return random.choice([0,1])      


def SN(iteration,initialp,stepsizes,probability):  
    iter = iteration
    initial = initialp
    stepsizes = stepsizes
    dis = []
    error = 0
    prob = probability
    x = initial
    w1 = initial
    w2 = initial
    traj = []
    traj.append(x)
    traj.append(x)
    for i in range(iter):
        hessinv = la.inv(hess_1(w1)+hess_2(w2))
        x = hessinv*(hess_1(w1)*w1-grad_1(w1)+hess_2(w2)*w2-grad_2(w2))

        if unisam1():
            w2 = x
        else:
            w1 = x
        
        traj.append(x)
        error = (grad_1(x)+grad_2(x)).transpose()*(grad_1(x)+grad_2(x))
        dis.append(error)
    return dis





iteration = 10
stepsize = 100**-1
p=0.3

MarinaRand=SN(iteration,innitialpoint,stepsize,p)
New = []
for i in MarinaRand:
    New.append(float(i))



plt.plot(range(iteration),New,linewidth = 0.5,label = 'SN with (a,b,c,d)=(50,300,200,7)')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error ')
#plt.title('Stepsize = 0.01')
plt.show()















def uniforms(p):

    tmp=random.uniform(0,1)
    if tmp<p:
        return 1
    else:
        return 0

def unisam():
    a = random.sample([0,1,2,3],2)
    d = matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    dd = matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    for i in a:
        dd[i,i]=0
    return d-dd

def top2(x):
    tmp = [abs(x[0,0]),abs(x[1,0]),abs(x[2,0]),abs(x[3,0])]
    lis = heapq.nlargest(2,range(len(tmp)),tmp.__getitem__)
    dd = matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    for i in lis:
        dd[i,i]=1
    return dd

def rank2(x):
    tmp = matrix([[x[0,0],x[1,0]],[x[2,0],x[3,0]]])
    dd=la.svd(tmp)
    tp = diag(dd[1])
    if abs(tp[0,0])>=abs(tp[1,1]):
        tp[1,1] = 0
    else:
        tp[0,0] = 0
    tm = dd[0]*tp*dd[2]
    return matrix([[tm[0,0]],[tm[0,1]],[tm[1,0]],[tm[1,1]]])