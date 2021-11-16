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

#define list of a b and calculate some key parameter

a = [matrix([[0.94884523, 0.31257516]]), matrix([[0.64695759, 0.79089169]]), matrix([[0.70218109, 0.91473775]]), matrix([[0.03035042, 0.21034799]]), matrix([[0.99278455, 0.2554682 ]]), matrix([[0.16064759, 0.09062056]]), matrix([[0.41438167, 0.77718962]]), matrix([[0.4953842 , 0.93027311]]), matrix([[0.7692516 , 0.19772597]]), matrix([[0.12430258, 0.03779965]])]
b = [matrix([[0.4231786]]), matrix([[0.524466]]), matrix([[0.17981303]]), matrix([[0.50033184]]), matrix([[0.71116473]]), matrix([[0.0604399]]), matrix([[0.37100656]]), matrix([[0.91263449]]), matrix([[0.66417151]]), matrix([[0.65995097]])]
a_T=[]
for i in a:
    a_T.append(i.T)
a_Ttimesa=matrix([[0,0],[0,0]])
a_Ttimesb=matrix([[0],[0]])
for i in range(10):
    a_Ttimesa = a_Ttimesa + (a_T[i])*a[i]
    a_Ttimesb = a_Ttimesb + (b[i])[0,0]*(a_T[i])

x_min = -(a_Ttimesa.I)*a_Ttimesb
print(x_min)


sigma = 3993.739347850591
tau_best = 10
tau = 10
prob = 0.2
                                       
 #define each function  and its gradient

class func:
    def __init__(self, a,b):
        self.a = a
        self.b = b
        
    def value(self,x):
        return  0.5*((self.a*x+self.b)[0,0])**2
                
    def grad(self,x):
        return ((self.a.T)*(self.a)*x+(self.a.T)*(self.b[0,0]))


funct = []
for i in range(10):
    funct.append(func(a[i],b[i]))              

#natural dithering gradient


#nice tau sampling
def standarddithering_grad(tauu,x,s):
    upper = np.ceil(np.abs(x)*s)/s
    dowm = np.floor(np.abs(x)*s)/s
    signn = np.sign(x)
    
    if np.abs(x)<s**-1:
        return 0
    
    tmp = random.uniform(0,1)
    if tmp< s*(np.abs(x)-dowm):
        return 1/tauu*upper*signn
    else:
        return 1/tauu*dowm*signn


def naturaldithering_grad(tauu,x,s):

    if np.abs(x)<2**(-(s-1)):
        return 0
    upper = 2**np.ceil(np.log2(np.abs(x)))
    dowm = 2**np.floor(np.log2(np.abs(x)))
    signn = np.sign(x)
    p = (upper-np.abs(x))/dowm
    tmp = random.uniform(0,1)
    if tmp<p:
        return 1/tauu*signn*dowm
    else:
        return 1/tauu*signn*upper


#hyper parameter
stepsize = 500**(-1)

init = matrix([[-0.5],[-0.2]])
x = 0
num_r = 10
dis = []

#iteration
#Natrual dithering
def expect(xx,yy,zz,dd,kk,jj,pp):
    tau = xx 
    iter = yy
    num_round = zz
    initial = dd
    min = kk
    stepsizes = jj
    dis = 0
    sss = pp

    for rd in range(num_round):
        x = initial
        traj = []
        traj.append(x)
        for i in range(iter):
            gradi = matrix([[0],[0]])
            for ff in range(10):
                normofgrad = ((funct[ff].grad(x)[0,0])**2+(funct[ff].grad(x)[1,0])**2)**(1/2)
                gradx = (funct[ff].grad(x)[0,0])/normofgrad
                grady = (funct[ff].grad(x)[1,0])/normofgrad
                gradi =  gradi + matrix([[naturaldithering_grad(tau,gradx,sss)*normofgrad],[naturaldithering_grad(tau,grady,sss)*normofgrad]])
               
            x = x - stepsizes*gradi   
            traj.append(x)
        dis = dis + (((x-min).T)*((x-min)))[0,0]
    return dis/num_round

#Standard dithering
def expect1(xx,yy,zz,dd,kk,jj,pp):  
    tau = xx 
    iter = yy
    num_round = zz
    initial = dd
    min = kk
    stepsizes = jj
    dis = 0
    sss = pp

    for rd in range(num_round):
        x = initial
        traj = []
        traj.append(x)
        for i in range(iter):
            gradi = matrix([[0],[0]])
            for ff in range(10):
                normofgrad = ((funct[ff].grad(x)[0,0])**2+(funct[ff].grad(x)[1,0])**2)**(1/2)
                gradx = (funct[ff].grad(x)[0,0])/normofgrad
                grady = (funct[ff].grad(x)[1,0])/normofgrad
                gradi =  gradi +  matrix([[standarddithering_grad(tau,gradx,sss)*normofgrad],[standarddithering_grad(tau,grady,sss)*normofgrad]])
               
            x = x - stepsizes*gradi   
            traj.append(x)
        dis = dis + (((x-min).T)*((x-min)))[0,0]
    return dis/num_round


dis_final =[]
dis_final1=[]
dis_final2=[]


iter = [100,500,1500,2000,2500]

for i in iter:
    print(i)
    dis_final.append(expect(10,i,num_r,init,x_min,stepsize,1))  #SGD
    print(i)
    dis_final1.append(expect(10,i,num_r,init,x_min,stepsize,3))#Sparsification
    print(i)
   # dis_final2.append(expect(10,i,num_r,init,x_min,stepsize,9))#Sparsification
    





#plot

plt.plot(iter,dis_final,linewidth = 1,marker= '*',markersize = 10,label = 's = 1')
plt.plot(iter,dis_final1,linewidth = 1,marker= '+',markersize = 10, label = 's = 3')
#plt.plot(iter,dis_final2,linewidth = 1,marker= 'x',markersize = 10, label = 's = 9')

 
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error ')
plt.show()