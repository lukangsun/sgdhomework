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
c=1
innitialpoint = matrix([[1],[1],[1],[1]])
#function
def f_1(x):
    return math.sin(x[0,0]+0.7)+math.sin(0.2*x[1,0]+0.5)+math.cos(0.1*x[3,0]+0.1)+math.cos(0.5*x[3,0]+0.4)

def f_2(x):
    return 0.5*(x[0,0]*x[0,0]+x[1,0]*x[1,0]+x[2,0]*x[2,0]+x[3,0]*x[3,0])

def grad_1(x):
    return (matrix([[math.cos(x[0,0]+0.7)],[0.2*math.cos(0.2*x[1,0]+0.5)],[-0.1*math.sin(0.1*x[2,0]+0.1)],[-0.5*math.sin(0.5*x[3,0]+0.4)]]))

def grad_2(x):
    return (matrix([[x[0,0]],[x[1,0]],[x[2,0]],[x[3,0]]]))

def unisam1():
    return random.choice([0,1,2,3,4,5,6,7,8,9])      

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


def MarinaRandk(iteration,initialp,stepsizes,probability):  
    iter = iteration
    initial = initialp
    stepsizes = stepsizes
    dis = []
    error = 0
    prob = probability
    x = initial
    g = grad_1(x)+grad_2(x)
    traj = []
    traj.append(x)
    traj.append(x)
    for i in range(iter):
        if uniforms(prob):
            x=x-stepsizes*(grad_1(x)+grad_2(x))
            g=grad_1(x)+grad_2(x)
        else:
            gradi = unisam()*(grad_1(x)-grad_1(traj[-2]))
            gradi = gradi+unisam()*(grad_2(x)-grad_2(traj[-2]))
            gradi = gradi+g
            g=gradi
            x = x - stepsizes*2*gradi
        traj.append(x)
        error = (grad_1(x)+grad_2(x)).transpose()*(grad_1(x)+grad_2(x))
        dis.append(error)
    return dis

def MarinaPermK(iteration,initialp,stepsizes,probability):
    iter = iteration
    initial = initialp
    stepsizes = stepsizes
    dis = []
    error = 0
    prob = probability
    x = initial
    g = grad_1(x)+grad_2(x)
    traj = []
    traj.append(x)
    traj.append(x)
    for i in range(iter):
        if uniforms(prob):
            x=x-stepsizes*(grad_1(x)+grad_2(x))
            g=grad_1(x)+grad_2(x)
        else:
            d = unisam()
            dd=matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])-d
            gradi = d*(grad_1(x)-grad_1(traj[-2]))
            gradi = gradi+dd*(grad_2(x)-grad_2(traj[-2]))
            gradi = gradi+g
            g=gradi
            x = x - stepsizes*2*gradi
        traj.append(x)
        error = (grad_1(x)+grad_2(x)).transpose()*(grad_1(x)+grad_2(x))
        dis.append(error)
    return dis

def EF21RANDK(iteration,initialp,stepsizes,probability):
    iter = iteration
    initial = initialp
    stepsizes = stepsizes
    dis = []
    error = 0
    prob = probability
    x = initial
    g = unisam()*(grad_1(x)+grad_2(x))
    traj = []
    traj.append(x)
    traj.append(x)
    for i in range(iter):
        d = unisam()
        gradi = d*(grad_1(x)+grad_2(x)-g)
        gradi = gradi+g
        g=gradi
        x = x - stepsizes*2*gradi
        traj.append(x)
        error = (grad_1(x)+grad_2(x)).transpose()*(grad_1(x)+grad_2(x))
        dis.append(error)
    return dis 


def EF21TOPK(iteration,initialp,stepsizes,probability):
    iter = iteration
    initial = initialp
    stepsizes = stepsizes
    dis = []
    error = 0
    prob = probability
    x = initial
    g = top2(grad_1(x)+grad_2(x))*(grad_1(x)+grad_2(x))
    traj = []
    traj.append(x)
    traj.append(x)
    for i in range(iter):
        gradi = top2(grad_1(x)+grad_2(x)-g)*(grad_1(x)+grad_2(x)-g)
        gradi = gradi+g
        g=gradi
        x = x - stepsizes*2*gradi
        traj.append(x)
        error = (grad_1(x)+grad_2(x)).transpose()*(grad_1(x)+grad_2(x))
        dis.append(error)
    return dis 

def EF21RANKK(iteration,initialp,stepsizes,probability):
    iter = iteration
    initial = initialp
    stepsizes = stepsizes
    dis = []
    error = 0
    prob = probability
    x = initial
    g = rank2(grad_1(x)+grad_2(x))
    traj = []
    traj.append(x)
    traj.append(x)
    for i in range(iter):
        gradi = rank2(grad_1(x)+grad_2(x)-g)
        gradi = gradi+g
        g=gradi
        x = x - stepsizes*2*gradi
        traj.append(x)
        error = (grad_1(x)+grad_2(x)).transpose()*(grad_1(x)+grad_2(x))
        dis.append(error)
    return dis 






iteration = 5
stepsize = 100**-1
p=0.3

MarinaRand=MarinaRandk(iteration,innitialpoint,stepsize,p)
New = []
for i in MarinaRand:
    New.append(float(i))

MarinaPerm=MarinaPermK(iteration,innitialpoint,stepsize,p)
Neew = []
for j in MarinaPerm:
    Neew.append(float(j))

EF21RAND=EF21RANDK(iteration,innitialpoint,stepsize,p)
EF = []
for k in EF21RAND:
    EF.append(float(k))

EF21Top=EF21TOPK(iteration,innitialpoint,stepsize,p)
EFT = []
for m in EF21Top:
    EFT.append(float(m))

EF21Rank = EF21RANDK(iteration,innitialpoint,stepsize,p)
EFTT = []
for n in EF21Rank:
    EFTT.append(float(n))


plt.plot(range(iteration),New,linewidth = 0.5,label = 'MARINA-RAND2')
plt.plot(range(iteration),Neew,linewidth = 0.5,label = 'MARINA-PERM2')
plt.plot(range(iteration),EF,linewidth = 0.5,label = 'EF21-RAND2')
plt.plot(range(iteration),EFT,linewidth = 0.5,label='EF21-TOP2')
plt.plot(range(iteration),EFTT,linewidth = 0.5, label = 'EF21-RANK1')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error ')
#plt.title('Stepsize = 0.01')
plt.show()